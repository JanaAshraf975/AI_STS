
import json
from pathlib import Path

from rag import answer


DATASET_PATH = "../evaluation/evaluation_dataset_arabic.json"

EXPERIMENTS = {
    "A": "solar_800_150",
    #"B": "solar_500_100",
    #"C": "solar_1000_200"
}

K = 5


def calculate_retrieval_metrics(retrieved_ids, relevant_ids):

    retrieved = set(retrieved_ids)
    relevant = set(relevant_ids)

    if not relevant:
        return 0, 0, 0

    true_positives = len(retrieved & relevant)

    precision = (
        true_positives / len(retrieved)
        if retrieved else 0
    )

    recall = true_positives / len(relevant)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return precision, recall, f1


def evaluate_collection(collection_name, dataset):

    results = []

    total_precision = 0
    total_recall = 0
    total_f1 = 0

    route_correct = 0
    hop_correct = 0

    for index, item in enumerate(dataset, start=1):

        question = item["question"]

        print(f"\nQuestion {index}: {question}")

        try:
            response = answer(
                question,
                k=K,
                collection_name=collection_name
            )

            retrieved_ids = response.get("chunk_ids", [])

            relevant_ids = item["relevant_chunk_ids"]

            precision, recall, f1 = calculate_retrieval_metrics(
                retrieved_ids,
                relevant_ids
            )

            route_match = (
                response.get("route") == item["expected_route"]
            )

            hop_match = (
                response.get("hop_type") == item["expected_hop_type"]
            )

            route_correct += int(route_match)
            hop_correct += int(hop_match)

            total_precision += precision
            total_recall += recall
            total_f1 += f1

            result = {
                "question": question,
                "expected_answer": item["expected_answer"],
                "generated_answer": response.get("answer", ""),
                "retrieved_chunk_ids": retrieved_ids,
                "relevant_chunk_ids": relevant_ids,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "expected_route": item["expected_route"],
                "actual_route": response.get("route"),
                "route_correct": route_match,
                "expected_hop_type": item["expected_hop_type"],
                "actual_hop_type": response.get("hop_type"),
                "hop_correct": hop_match
            }

            results.append(result)

            print(f"Precision: {precision:.3f}")
            print(f"Recall:    {recall:.3f}")
            print(f"F1:        {f1:.3f}")

        except Exception as e:
            print(f"Error: {e}")

    total = len(results)

    if total == 0:
        return {
            "collection": collection_name,
            "error": "No questions were evaluated"
        }

    summary = {
        "collection": collection_name,
        "total_questions": total,
        "precision": total_precision / total,
        "recall": total_recall / total,
        "f1": total_f1 / total,
        "route_accuracy": route_correct / total,
        "hop_type_accuracy": hop_correct / total,
        "details": results
    }

    return summary


def main():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    all_results = {}

    for experiment, collection_name in EXPERIMENTS.items():

        print("\n" + "=" * 50)
        print(f"EXPERIMENT {experiment}: {collection_name}")
        print("=" * 50)

        summary = evaluate_collection(
            collection_name,
            dataset
        )

        all_results[experiment] = summary

        print("\nRESULTS:")
        print(f"Precision: {summary.get('precision', 0):.3f}")
        print(f"Recall:    {summary.get('recall', 0):.3f}")
        print(f"F1:        {summary.get('f1', 0):.3f}")

    with open(
        "arabic_evaluation_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_results,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\nEvaluation completed!")
    print("Results saved to arabic_evaluation_results.json")


if __name__ == "__main__":
    main()
