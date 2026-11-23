
import json
import math
from pathlib import Path
from statistics import mean

from vectoredb import search


DATASET_FILE = "../evaluation/eval_dataset.json"

# Evaluate different retrieval depths
K_VALUES = [5]


def precision_at_k(retrieved, relevant, k):

    if k == 0:
        return 0.0

    hits = sum(
        1 for chunk_id in retrieved[:k]
        if chunk_id in relevant
    )

    return hits / k


def recall_at_k(retrieved, relevant, k):

    if not relevant:
        return 0.0

    hits = sum(
        1 for chunk_id in retrieved[:k]
        if chunk_id in relevant
    )

    return hits / len(relevant)


def reciprocal_rank(retrieved, relevant):

    for rank, chunk_id in enumerate(retrieved, start=1):

        if chunk_id in relevant:
            return 1 / rank

    return 0.0


def hit_rate_at_k(retrieved, relevant, k):

    return float(
        any(
            chunk_id in relevant
            for chunk_id in retrieved[:k]
        )
    )


def evaluate_one(question, relevant_ids, k):

    results = search(
        question,
        k=k,
        threshold=0.6
    )

    retrieved_ids = [
        chunk["id"]
        for chunk in results
    ]
    print("Retrieved count:", len(retrieved_ids))

    relevant = set(relevant_ids)

    return {
        "precision": precision_at_k(
            retrieved_ids, relevant, k
        ),
        "recall": recall_at_k(
            retrieved_ids, relevant, k
        ),
        "mrr": reciprocal_rank(
            retrieved_ids, relevant
        ),
        "hit_rate": hit_rate_at_k(
            retrieved_ids, relevant, k
        ),
        "retrieved_ids": retrieved_ids,
        "relevant_ids": list(relevant)
    }


def evaluate_dataset():

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        dataset = json.load(file)

    questions = dataset["data"]

    if not questions:
        print("Evaluation dataset is empty.")
        return

    final_results = {}

    for k in K_VALUES:

        print(f"\nEvaluating at K={k}")

        question_results = []

        for index, item in enumerate(questions, start=1):

            result = evaluate_one(
                item["question"],
                item["relevant_chunk_ids"],
                k
            )

            question_results.append(result)

            print(
                f"Question {index}/{len(questions)} | "
                f"P@{k}: {result['precision']:.3f} | "
                f"R@{k}: {result['recall']:.3f}"
            )

        final_results[f"top_{k}"] = {
            "precision_at_k": mean(
                r["precision"]
                for r in question_results
            ),
            "recall_at_k": mean(
                r["recall"]
                for r in question_results
            ),
            "mrr": mean(
                r["mrr"]
                for r in question_results
            ),
            "hit_rate_at_k": mean(
                r["hit_rate"]
                for r in question_results
            ),
            "details": question_results
        }

    print("\n========== FINAL RESULTS ==========")

    for k, metrics in final_results.items():

        print(f"\n{k.upper()}")

        for metric in [
            "precision_at_k",
            "recall_at_k",
            "mrr",
            "hit_rate_at_k"
        ]:

            print(
                f"{metric}: "
                f"{metrics[metric]:.4f}"
            )

    with open(
        "retrieval_evaluation_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_results,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        "\nResults saved to retrieval_evaluation_results.json"
    )


if __name__ == "__main__":
    evaluate_dataset()
