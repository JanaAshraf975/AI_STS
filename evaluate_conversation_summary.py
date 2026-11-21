
import json

from rag import update_conversation_summary, parse_json
from llm import ask_llm, JUDGE_MODEL


# =====================================
# Test Cases
# =====================================

TEST_CASES = [

    {
        "name": "TC1 - Basic Summary",

        "previous_summary": "",

        "old_messages": [
            {
                "role": "user",
                "content": "I am building an Adaptive RAG project."
            },
            {
                "role": "assistant",
                "content": "That's a project that combines retrieval and generation."
            },
            {
                "role": "user",
                "content": "I use ChromaDB to store document embeddings."
            },
            {
                "role": "assistant",
                "content": "ChromaDB can store and retrieve vector embeddings."
            },
            {
                "role": "user",
                "content": "I also use BGE embeddings."
            },
            {
                "role": "assistant",
                "content": "BGE models convert text into vector representations."
            }
        ],

        "expected_facts": [
            "The user is building an Adaptive RAG project.",
            "The user uses ChromaDB.",
            "The user uses BGE embeddings."
        ]
    },


    {
        "name": "TC2 - User Preferences",

        "previous_summary": "",

        "old_messages": [
            {
                "role": "user",
                "content": "I prefer explanations in simple English."
            },
            {
                "role": "assistant",
                "content": "I will use simple and clear explanations."
            },
            {
                "role": "user",
                "content": "Please explain things step by step."
            },
            {
                "role": "assistant",
                "content": "Sure, I will break topics into smaller steps."
            },
            {
                "role": "user",
                "content": "I like practical examples and code examples."
            },
            {
                "role": "assistant",
                "content": "I will include practical examples when explaining concepts."
            }
        ],

        "expected_facts": [
            "The user prefers simple English explanations.",
            "The user prefers step-by-step explanations.",
            "The user likes practical examples.",
            "The user likes code examples."
        ]
    },


    {
        "name": "TC3 - Important Technical Details",

        "previous_summary": "",

        "old_messages": [
            {
                "role": "user",
                "content": "My backend API uses the endpoint /api/v1/search."
            },
            {
                "role": "assistant",
                "content": "Okay, the search endpoint is /api/v1/search."
            },
            {
                "role": "user",
                "content": "I use Redis for caching."
            },
            {
                "role": "assistant",
                "content": "Redis can cache frequently requested data."
            },
            {
                "role": "user",
                "content": "The cache TTL is 3600 seconds."
            },
            {
                "role": "assistant",
                "content": "That means cached data expires after 3600 seconds."
            }
        ],

        "expected_facts": [
            "The API endpoint is /api/v1/search.",
            "The project uses Redis for caching.",
            "The cache TTL is 3600 seconds."
        ]
    },


    {
        "name": "TC4 - Previous Summary Preservation",

        "previous_summary": (
            "The user is building an Adaptive RAG project. "
            "The project uses ChromaDB and BGE embeddings."
        ),

        "old_messages": [
            {
                "role": "user",
                "content": "I added FastAPI to my project."
            },
            {
                "role": "assistant",
                "content": "FastAPI can be used to create the API."
            },
            {
                "role": "user",
                "content": "I also added Redis for caching."
            },
            {
                "role": "assistant",
                "content": "Redis can improve response time through caching."
            }
        ],

        "expected_facts": [
            "The user is building an Adaptive RAG project.",
            "The project uses ChromaDB.",
            "The project uses BGE embeddings.",
            "The project uses FastAPI.",
            "The project uses Redis for caching."
        ]
    },

    {
        "name": "TC5 - Conflicting Information",
        "previous_summary": (
            "The project uses Redis for caching. "
            "The cache TTL is 3600 seconds."
        ),
        "old_messages": [
            {
                "role": "user",
                "content": (
                    "We changed the Redis cache TTL "
                    "from 3600 seconds to 1800 seconds."
                )
            },
            {
                "role": "assistant",
                "content": (
                    "Understood. The new cache TTL is "
                    "1800 seconds."
                )
            }
        ],
        "expected_facts": [
            "The project uses Redis for caching.",
            "The new cache TTL is 1800 seconds."
        ]
    },
    {
        "name": "TC6 - Long Conversation",
        "previous_summary": (
            "The project is an Adaptive RAG application "
            "using ChromaDB."
        ),
        "old_messages": [
            {
                "role": "user",
                "content": "We use BGE embeddings."
            },
            {
                "role": "assistant",
                "content": "Okay, BGE embeddings are used."
            },
            {
                "role": "user",
                "content": "Our backend is built with FastAPI."
            },
            {
                "role": "assistant",
                "content": "Got it, the backend uses FastAPI."
            },
            {
                "role": "user",
                "content": "We use Redis for caching."
            },
            {
                "role": "assistant",
                "content": "Redis will handle caching."
            },
            {
                "role": "user",
                "content": "The cache TTL is 1800 seconds."
            },
            {
                "role": "assistant",
                "content": "Understood."
            },
            {
                "role": "user",
                "content": "We use cosine similarity for retrieval."
            },
            {
                "role": "assistant",
                "content": "Okay."
            }
        ],
        "expected_facts": [
            "The project is an Adaptive RAG application.",
            "The project uses ChromaDB.",
            "The project uses BGE embeddings.",
            "The backend uses FastAPI.",
            "The project uses Redis for caching.",
            "The cache TTL is 1800 seconds.",
            "The project uses cosine similarity for retrieval."
        ]
    },
    {
        "name": "TC7 - Important Detail Preservation",
        "previous_summary": (
            "The project uses FastAPI and Redis."
        ),
        "old_messages": [
            {
                "role": "user",
                "content": (
                    "Our search endpoint is /api/v2/search."
                )
            },
            {
                "role": "assistant",
                "content": "Okay."
            },
            {
                "role": "user",
                "content": (
                    "The endpoint accepts POST requests "
                    "and returns JSON responses."
                )
            },
            {
                "role": "assistant",
                "content": "Understood."
            }
        ],
        "expected_facts": [
            "The project uses FastAPI.",
            "The project uses Redis.",
            "The search endpoint is /api/v2/search.",
            "The endpoint accepts POST requests.",
            "The endpoint returns JSON responses."
        ]
    }


]


# =====================================
# Summary Evaluation
# =====================================

def evaluate_summary(
    previous_summary,
    old_messages,
    generated_summary,
    expected_facts
):

    conversation = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in old_messages
    )

    prompt = f"""
You are an evaluator for a conversation summarization system.

Your task is to evaluate the generated summary.

Use BOTH the previous summary and the new conversation
as the source of truth.

Previous Summary:
{previous_summary}

New Conversation:
{conversation}

Expected Facts:
{json.dumps(expected_facts, ensure_ascii=False)}

Generated Summary:
{generated_summary}

Evaluate the generated summary.

1. preserved_facts:
   Expected facts that are correctly preserved.

2. missing_facts:
   Expected facts that are missing or incorrectly represented.

3. hallucinated_facts:
   Claims in the generated summary that are not supported
   by the previous summary or new conversation.

4. summary_fact_count:
   Total number of factual claims in the generated summary.

Return valid JSON only:

{{
    "preserved_facts": [],
    "missing_facts": [],
    "hallucinated_facts": [],
    "summary_fact_count": 0
}}
"""

    result = ask_llm(
        prompt,
        model=JUDGE_MODEL,
        temperature=0,
        json_mode=True,
    )

    return parse_json(result)


# =====================================
# Run Evaluation
# =====================================

def run_evaluation():

    all_results = []

    for test_case in TEST_CASES:

        print("\n" + "=" * 50)
        print(test_case["name"])
        print("=" * 50)

        previous_summary = test_case["previous_summary"]
        old_messages = test_case["old_messages"]
        expected_facts = test_case["expected_facts"]

        # Generate Summary
        generated_summary = update_conversation_summary(
            previous_summary,
            old_messages
        )

        print("\nGenerated Summary:")
        print(generated_summary)

        # Evaluate Summary using Judge Model
        evaluation = evaluate_summary(
            previous_summary,
            old_messages,
            generated_summary,
            expected_facts
        )

        preserved = evaluation.get("preserved_facts", [])
        missing = evaluation.get("missing_facts", [])
        hallucinated = evaluation.get("hallucinated_facts", [])

        # Calculate Metrics
        retention_rate = (
            len(preserved) / len(expected_facts) * 100
            if expected_facts else 0
        )

        summary_fact_count = evaluation.get(
            "summary_fact_count", 0
        )

        hallucination_rate = (
            len(hallucinated) / summary_fact_count * 100
            if summary_fact_count else 0
        )

        print("\nEvaluation Results:")
        print("Preserved Facts:", preserved)
        print("Missing Facts:", missing)
        print("Hallucinated Facts:", hallucinated)

        print(f"\nFact Retention Rate: {retention_rate:.2f}%")
        print(f"Hallucination Rate: {hallucination_rate:.2f}%")

        all_results.append({
            "test_case": test_case["name"],
            "generated_summary": generated_summary,
            "evaluation": evaluation,
            "fact_retention_rate": round(retention_rate, 2),
            "hallucination_rate": round(hallucination_rate, 2)
        })

    # Save Results
    with open(
        "conversation_summary_evaluation_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 50)
    print("Evaluation Completed Successfully!")
    print("Results saved to conversation_summary_evaluation_results.json")


if __name__ == "__main__":
    run_evaluation()
