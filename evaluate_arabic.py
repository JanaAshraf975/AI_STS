
import json

from rag import answer


DATASET_PATH = "../evaluation/evaluation_dataset_longArabic.json"
OUTPUT_PATH = "arabic_evaluation_results.json"

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


def evaluate(dataset):

    results = []

    total_precision = 0
    total_recall = 0
    total_f1 = 0

    for index, item in enumerate(dataset, start=1):

        question = item["question"]

        print(f"\nQuestion {index}: {question}")

        try:
            response = answer(
                question,
                k=K
            )

            retrieved_ids = response.get("chunk_ids", [])
            relevant_ids = item.get("relevant_chunk_ids", [])

            precision, recall, f1 = calculate_retrieval_metrics(
                retrieved_ids,
                relevant_ids
            )

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
                "f1": f1
            }

            results.append(result)

            print(f"Precision: {precision:.3f}")
            print(f"Recall:    {recall:.3f}")
            print(f"F1:        {f1:.3f}")

        except Exception as e:
            print(f"Error: {e}")

    total = len(results)

    if total == 0:
        print("No questions were evaluated.")
        return

    summary = {
        "total_questions": total,
        "precision": total_precision / total,
        "recall": total_recall / total,
        "f1": total_f1 / total,
        "details": results
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)

    print(f"Total Questions: {total}")
    print(f"Precision: {summary['precision']:.3f}")
    print(f"Recall:    {summary['recall']:.3f}")
    print(f"F1:        {summary['f1']:.3f}")

    print(f"\nResults saved to: {OUTPUT_PATH}")


def main():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    evaluate(dataset)


if __name__ == "__main__":
    main()
