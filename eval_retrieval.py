
import json
from statistics import mean

from vectoredb import search


DATASET_FILE = "../evaluation/eval2_dataset.json"

K_VALUES = [3]
THRESHOLD = 0.6


def remove_duplicates(items):
    """Remove duplicate IDs while preserving their order."""
    return list(dict.fromkeys(items))


def precision_at_k(retrieved, relevant, k):
    retrieved = remove_duplicates(retrieved[:k])

    if not retrieved:
        return 0.0

    hits = sum(
        1 for chunk_id in retrieved
        if chunk_id in relevant
    )

    return hits / len(retrieved)


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0

    retrieved = set(retrieved[:k])

    hits = len(retrieved.intersection(relevant))

    return hits / len(relevant)


def reciprocal_rank(retrieved, relevant):
    retrieved = remove_duplicates(retrieved)

    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1 / rank

    return 0.0


def hit_rate_at_k(retrieved, relevant, k):
    retrieved = set(retrieved[:k])

    return float(
        bool(retrieved.intersection(relevant))
    )


def evaluate_one(question, relevant_ids, k):

    results = search(
        question,
        k=k,
        threshold=THRESHOLD
    )

    retrieved_ids = [
        chunk["id"]
        for chunk in results
    ]

    # Remove duplicate retrieved IDs
    retrieved_ids = remove_duplicates(retrieved_ids)

    relevant = set(relevant_ids)

    return {
        "is_answerable": bool(relevant),

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

        "retrieved_count": len(retrieved_ids),

        "retrieved_ids": retrieved_ids,

        "relevant_ids": list(relevant)
    }


def safe_mean(values):
    return mean(values) if values else 0.0


def evaluate_dataset():

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        dataset = json.load(file)

    if not dataset:
        print("Evaluation dataset is empty.")
        return

    final_results = {}

    for k in K_VALUES:

        print(f"\n========== EVALUATING TOP {k} ==========")

        question_results = []

        for index, item in enumerate(dataset, start=1):

            result = evaluate_one(
                item["question"],
                item["relevant_chunk_ids"],
                k
            )

            question_results.append(result)

            status = (
                "Answerable"
                if result["is_answerable"]
                else "Unanswerable"
            )

            print(
                f"Question {index}/{len(dataset)} | "
                f"{status} | "
                f"Retrieved: {result['retrieved_count']} | "
                f"P: {result['precision']:.3f} | "
                f"R: {result['recall']:.3f}"
            )

        # Separate answerable and unanswerable questions
        answerable = [
            r for r in question_results
            if r["is_answerable"]
        ]

        unanswerable = [
            r for r in question_results
            if not r["is_answerable"]
        ]

        # Answerable metrics
        answerable_metrics = {
            "total_questions": len(answerable),

            "precision": safe_mean([
                r["precision"] for r in answerable
            ]),

            "recall": safe_mean([
                r["recall"] for r in answerable
            ]),

            "mrr": safe_mean([
                r["mrr"] for r in answerable
            ]),

            "hit_rate": safe_mean([
                r["hit_rate"] for r in answerable
            ]),

            "questions_with_results": sum(
                bool(r["retrieved_count"])
                for r in answerable
            ),

            "questions_without_results": sum(
                not r["retrieved_count"]
                for r in answerable
            )
        }

        # Unanswerable metrics
        correctly_empty = sum(
            not r["retrieved_count"]
            for r in unanswerable
        )

        false_results = sum(
            bool(r["retrieved_count"])
            for r in unanswerable
        )

        unanswerable_metrics = {
            "total_questions": len(unanswerable),

            "correctly_empty": correctly_empty,

            "false_results": false_results,

            "correct_empty_rate": (
                correctly_empty / len(unanswerable)
                if unanswerable else 0.0
            )
        }

        final_results[f"top_{k}"] = {
            "threshold": THRESHOLD,

            "answerable": answerable_metrics,

            "unanswerable": unanswerable_metrics,

            "details": question_results
        }

    # Print final results
    print("\n========== FINAL RESULTS ==========")

    for k, metrics in final_results.items():

        print(f"\n{k.upper()}")

        print("\n--- ANSWERABLE QUESTIONS ---")

        answerable = metrics["answerable"]

        print(
            f"Total Questions: "
            f"{answerable['total_questions']}"
        )

        print(
            f"Precision: {answerable['precision']:.4f}"
        )

        print(
            f"Recall: {answerable['recall']:.4f}"
        )

        print(
            f"MRR: {answerable['mrr']:.4f}"
        )

        print(
            f"Hit Rate: {answerable['hit_rate']:.4f}"
        )

        print(
            f"Questions With Results: "
            f"{answerable['questions_with_results']}"
        )

        print(
            f"Questions Without Results: "
            f"{answerable['questions_without_results']}"
        )

        print("\n--- UNANSWERABLE QUESTIONS ---")

        unanswerable = metrics["unanswerable"]

        print(
            f"Total Questions: "
            f"{unanswerable['total_questions']}"
        )

        print(
            f"Correctly Empty: "
            f"{unanswerable['correctly_empty']}"
        )

        print(
            f"False Results: "
            f"{unanswerable['false_results']}"
        )

        print(
            f"Correct Empty Rate: "
            f"{unanswerable['correct_empty_rate']:.4f}"
        )

    # Save results
    output_file = "retrieval_evaluation_results.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_results,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    evaluate_dataset()
