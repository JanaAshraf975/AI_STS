
import json
from llm import ask_llm, GENERATOR_MODEL, JUDGE_MODEL
from rag import SYSTEM, build_prompt


test_cases = [
    {
        "name": "Answerable question",
        "question": "How much does DoorGuard Sensor cost?",
        "context": [
            {
                "source": "Product_Catalog.pdf",
                "page": 7,
                "text": "The DoorGuard Sensor costs EGP 899."
            }
        ],
        "expected_behavior": "answer"
    },
    {
        "name": "Unanswerable question",
        "question": "Does DoorGuard support Bluetooth?",
        "context": [
            {
                "source": "Product_Catalog.pdf",
                "page": 7,
                "text": "The DoorGuard Sensor costs EGP 899."
            }
        ],
        "expected_behavior": "fallback"
    },
    {
        "name": "Missing product",
        "question": "How much does SmartLock X cost?",
        "context": [
            {
                "source": "Product_Catalog.pdf",
                "page": 7,
                "text": "The DoorGuard Sensor costs EGP 899."
            }
        ],
        "expected_behavior": "fallback"
    },
    {
        "name": "Wrong information in context",
        "question": "How much does DoorGuard cost?",
        "context": [
            {
                "source": "Product_Catalog.pdf",
                "page": 5,
                "text": "The ViewCam Indoor costs EGP 2499."
            }
        ],
        "expected_behavior": "fallback"
    }
]


def judge_prompt_compliance(question, context, response):

    context_text = "\n\n".join(
        f"[{i}] ({chunk['source']} p.{chunk['page']})\n{chunk['text']}"
        for i, chunk in enumerate(context, start=1)
    )

    judge_prompt = f"""
You are evaluating whether a RAG Generator follows its SYSTEM instructions.

SYSTEM INSTRUCTIONS:
{SYSTEM}

Question:
{question}

Retrieved Context:
{context_text}

Generator Answer:
{response}

Evaluate the answer according to these rules:

1. Context compliance:
   Does the answer use ONLY information supported by the provided context?

2. Factual correctness:
   Does the answer correctly answer the question without incorrect or contradictory claims?

3. Citation compliance:
   Are citations included where needed, and does each citation refer to a context chunk
   that actually supports the claim?

4. Instruction compliance:
   Does the answer follow the SYSTEM instructions, including the required fallback
   when the context does not contain the answer?

Return JSON only:

{{
    "context_compliance": true,
    "factual_correctness": true,
    "citation_compliance": true,
    "instruction_compliance": true,
    "reason": "Short explanation"
}}

Set a field to false if its condition is not satisfied.
Do not accept unsupported information.
"""

    result = ask_llm(
        judge_prompt,
        model=JUDGE_MODEL,
        temperature=0,
        json_mode=True
    )

    return json.loads(result)


passed_count = 0

for test in test_cases:

    prompt = build_prompt(
        test["question"],
        test["context"]
    )

    response = ask_llm(
        prompt,
        model=GENERATOR_MODEL,
        system=SYSTEM,
        temperature=0
    )

    expected = test["expected_behavior"]

    print(f"\nTest: {test['name']}")
    print(f"Expected behavior: {expected}")
    print(f"Generated answer: {response}")

    if expected == "fallback":

        passed = (
            response.strip()
            == "I could not find this in the documents."
        )

        print(
            "Fallback compliance:",
            "PASS" if passed else "FAIL"
        )

    else:

        try:
            evaluation = judge_prompt_compliance(
                test["question"],
                test["context"],
                response
            )

            checks = [
                evaluation["context_compliance"],
                evaluation["factual_correctness"],
                evaluation["citation_compliance"],
                evaluation["instruction_compliance"]
            ]

            for name, result in evaluation.items():
                if name != "reason":
                    print(
                        f"{name}:",
                        "PASS" if result else "FAIL"
                    )

            print("Judge reason:", evaluation["reason"])

            passed = all(checks)

        except (ValueError, KeyError, TypeError) as e:
            print("Judge evaluation error:", e)
            passed = False

    if passed:
        passed_count += 1

    print(f"Final result: {'PASS' if passed else 'FAIL'}")


total = len(test_cases)
failed_count = total - passed_count

print("\n========== PROMPT EVALUATION ==========")
print(f"Passed: {passed_count}/{total}")
print(f"Failed: {failed_count}")
print(f"Prompt Compliance: {passed_count / total:.3f}")
