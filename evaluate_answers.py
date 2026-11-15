import json
from rag import answer
from llm import ask_llm, JUDGE_MODEL


DATASET_PATH = "../evaluation/evaluation_dataset.json"


def evaluate_answer(question, expected_answer, generated_answer, contexts):

    context_text = "\n\n".join(contexts) if contexts else "No PDF context available."

    prompt = f"""
You are an evaluator for a Retrieval-Augmented Generation (RAG) system.

Evaluate the generated answer using the question, expected answer,
and retrieved context.

Question:
{question}

Expected Answer:
{expected_answer}

Generated Answer:
{generated_answer}

Retrieved Context:
{context_text}

Evaluate these three metrics:

1. correctness:
Compare the generated answer with the expected answer.
A score of 1 means fully correct.
A score of 0 means completely incorrect.
Partial scores are allowed.

2. faithfulness:
Check whether the generated answer is supported by the retrieved context.
Do not reward information that is not supported by the context.
If no PDF context is available, return null.

3. relevancy:
Check whether the generated answer directly answers the question.
A score of 1 means fully relevant.
A score of 0 means irrelevant.

Return JSON only in this format:

{{
    "correctness": 0.0,
    "faithfulness": 0.0,
    "relevancy": 0.0,
    "reason": "Brief explanation"
}}
"""

    result = ask_llm(
        prompt,
        model=JUDGE_MODEL,
        temperature=0,
        json_mode=True,
    )

    return json.loads(result)


with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)


correctness_scores = []
faithfulness_scores = []
relevancy_scores = []

for item in dataset:

    question = item["question"]
    expected_answer = item["expected_answer"]

    if not item.get("answerable", True):
        print(f"\nSKIPPED: {question} (not answerable)")
        continue

    # Generate the actual RAG answer
    result = answer(question, k=5, history=[])

    generated_answer = result["answer"]
    contexts = result["contexts"]

    # Evaluate the generated answer
    scores = evaluate_answer(
        question,
        expected_answer,
        generated_answer,
        contexts,
    )

    correctness = float(scores["correctness"])
    faithfulness = scores["faithfulness"]
    relevancy = float(scores["relevancy"])

    correctness_scores.append(correctness)
    relevancy_scores.append(relevancy)

    if faithfulness is not None:
        faithfulness = float(faithfulness)
        faithfulness_scores.append(faithfulness)

    print("\n" + "=" * 60)
    print("Question:", question)
    print("\nExpected Answer:", expected_answer)
    print("\nGenerated Answer:", generated_answer)

    print("\nRoute:", result["route"])
    print("Hop Type:", result["hop_type"])
    print("Retried:", result["retried"])

    print("\nCorrectness:", correctness)
    print("Faithfulness:", faithfulness)
    print("Relevancy:", relevancy)
    print("Reason:", scores["reason"])


def average(scores):
    return sum(scores) / len(scores) if scores else 0


print("\n" + "=" * 60)
print("===== FINAL ANSWER EVALUATION RESULTS =====")

print(f"Average Correctness: {average(correctness_scores):.3f}")
print(f"Average Faithfulness: {average(faithfulness_scores):.3f}")
print(f"Average Relevancy: {average(relevancy_scores):.3f}")

print("\nEvaluated Questions:", len(correctness_scores))