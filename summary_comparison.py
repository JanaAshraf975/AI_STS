import json

from rag import answer, build_prompt, SYSTEM
from llm import ask_llm, GENERATOR_MODEL, JUDGE_MODEL


DATASET_PATH = "../evaluation/evaluation_dataset.json"


# ==========================================
# 1. Get Ground Truth from Dataset
# ==========================================

def get_ground_truth(item):

    possible_keys = [
        "expected_answer",
        "ground_truth",
        "answer"
    ]

    for key in possible_keys:
        if item.get(key):
            return item[key]

    raise KeyError(
        "No Ground Truth field found. "
        "Expected one of: expected_answer, ground_truth, answer"
    )


# ==========================================
# 2. LLM-as-a-Judge Evaluation
# ==========================================

def judge_answer(question, chunks, summaries, response, ground_truth):

    chunk_context = "\n\n".join(
        f"[{i}]\n{c['text'] if isinstance(c, dict) else c}"
        for i, c in enumerate(chunks, start=1)
    )

    summary_context = "\n\n".join(
        f"[S{i}] PDF Summary:\n"
        f"{s['text'] if isinstance(s, dict) else s}"
        for i, s in enumerate(summaries, start=1)
    )

    prompt = f"""
You are an expert evaluator for a Retrieval-Augmented Generation system.

Question:
{question}

Ground Truth Answer:
{ground_truth}

Original Retrieved Chunks:
{chunk_context}

PDF Summaries:
{summary_context if summary_context else "No summaries provided."}

Generator Answer:
{response}

Evaluate the Generator Answer using the following criteria:

1. factual_correctness:
Compare the answer with the Ground Truth.
Set true if the answer contains the correct facts.
Accept different wording if the meaning is correct.
Set false if the answer gives incorrect information or misses a critical fact.

2. faithfulness:
Check whether the answer is supported by the provided Original Retrieved Chunks
or PDF Summaries.
Set false if it contains unsupported or invented information.

3. answer_relevance:
Check whether the answer directly answers the question.
Set false if the answer is irrelevant or refuses to answer despite sufficient
information being available in the context.

4. citation_compliance:
Check whether citations are present where needed, refer to available sources,
and support the claims.
Do not mark true just because citation numbers exist.

Important:
- Do not require exact wording from the Ground Truth.
- Do not accept unsupported claims.
- A refusal is incorrect if the provided context contains the answer.
- Evaluate only the answer provided, not what the model might have intended.

Return JSON only:
{{
    "factual_correctness": true,
    "faithfulness": true,
    "answer_relevance": true,
    "citation_compliance": true,
    "reason": "Short explanation"
}}
"""

    result = ask_llm(
        prompt,
        model=JUDGE_MODEL,
        temperature=0,
        json_mode=True,
        max_tokens=500
    )

    return json.loads(result)


# ==========================================
# 3. Load Evaluation Dataset
# ==========================================

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)


baseline_scores = []
summary_scores = []

successful_questions = 0
failed_questions = 0


# ==========================================
# 4. Run Summary Comparison
# ==========================================

for index, item in enumerate(dataset, start=1):

    question = item["question"]
    ground_truth = get_ground_truth(item)

    print("\n" + "=" * 60)
    print(f"Question {index}: {question}")

    print("\nGround Truth:")
    print(ground_truth)

    try:

        # Run Adaptive RAG once to get final chunks and summaries
        result = answer(
            question,
            k=5,
            history=[],
            conversation_summary=""
        )

        chunks = result["contexts"]
        summaries = result["summaries"]

        print("\nChunks used:", len(chunks))
        print("Summaries used:", len(summaries))

        # ----------------------------------
        # Experiment A: Baseline
        # Chunks only
        # ----------------------------------

        baseline_response = ask_llm(
            build_prompt(question, chunks, summaries=[]),
            model=GENERATOR_MODEL,
            system=SYSTEM,
            temperature=0,
            max_tokens=800
        )

        # ----------------------------------
        # Experiment B: New System
        # Chunks + Summary
        # ----------------------------------

        summary_response = result["answer"]

        # ----------------------------------
        # Evaluate Baseline
        # ----------------------------------

        baseline_eval = judge_answer(
            question,
            chunks,
            [],
            baseline_response,
            ground_truth
        )

        # ----------------------------------
        # Evaluate New System
        # ----------------------------------

        summary_eval = judge_answer(
            question,
            chunks,
            summaries,
            summary_response,
            ground_truth
        )

        # ----------------------------------
        # Print Results
        # ----------------------------------

        print("\n--- BASELINE: CHUNKS ONLY ---")
        print(baseline_response)
        print(baseline_eval)

        print("\n--- NEW SYSTEM: CHUNKS + SUMMARY ---")
        print(summary_response)
        print(summary_eval)

        baseline_scores.append(baseline_eval)
        summary_scores.append(summary_eval)

        successful_questions += 1

    except Exception as e:

        print("\nQuestion failed:", question)
        print("Error:", str(e))

        failed_questions += 1

        continue


# ==========================================
# 5. Calculate Final Scores
# ==========================================

def calculate_scores(results):

    metrics = [
        "factual_correctness",
        "faithfulness",
        "answer_relevance",
        "citation_compliance"
    ]

    scores = {}

    for metric in metrics:

        scores[metric] = (
            sum(
                1 for result in results
                if result.get(metric) is True
            ) / len(results)
            if results else 0
        )

    return scores


baseline_final = calculate_scores(baseline_scores)
summary_final = calculate_scores(summary_scores)


# ==========================================
# 6. Final Comparison
# ==========================================

print("\n" + "=" * 60)
print("========== FINAL COMPARISON ==========")

print("\nSuccessful questions:", successful_questions)
print("Failed questions:", failed_questions)

print("\nEvaluated answers:", len(baseline_scores))

for metric in baseline_final:

    print(f"\n{metric}:")

    print(f"Baseline:     {baseline_final[metric]:.3f}")
    print(f"With Summary: {summary_final[metric]:.3f}")

    difference = summary_final[metric] - baseline_final[metric]

    print(f"Difference:   {difference:+.3f}")

print("\n======================================")