
import json
from rag import answer

DATASET_PATH = "../evaluation/evaluation_dataset.json"


with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)


total_precision = 0
total_recall = 0
total_coverage = 0

evaluated_questions = 0
no_context_questions = 0


for item in dataset:

    if not item.get("answerable", True):
        continue

    question = item["question"]
    relevant_ids = set(item["relevant_chunk_ids"])

    # Get the final context used by the Generator
    result = answer(question, k=5, history=[])

    retrieved_ids = result["chunk_ids"]

    # No PDF context was passed to the Generator
    if not retrieved_ids:
        no_context_questions += 1
        continue

    retrieved_ids = set(retrieved_ids)

    relevant_found = len(retrieved_ids & relevant_ids)

    precision = relevant_found / len(retrieved_ids)
    recall = relevant_found / len(relevant_ids) if relevant_ids else 0

    # Did we retrieve all expected relevant chunks?
    coverage = 1 if relevant_ids.issubset(retrieved_ids) else 0

    total_precision += precision
    total_recall += recall
    total_coverage += coverage

    evaluated_questions += 1

    print(f"\nQuestion: {question}")
    print(f"Route: {result['route']}")
    print(f"Retrieved IDs: {len(retrieved_ids)}")
    print(f"Relevant found: {relevant_found}")
    print(f"Context Precision: {precision:.3f}")
    print(f"Context Recall: {recall:.3f}")
    print(f"Context Coverage: {coverage}")


if evaluated_questions > 0:
    print("\n========== CONTEXT EVALUATION ==========")

    print(
        f"Average Context Precision: "
        f"{total_precision / evaluated_questions:.3f}"
    )

    print(
        f"Average Context Recall: "
        f"{total_recall / evaluated_questions:.3f}"
    )

    print(
        f"Average Context Coverage: "
        f"{total_coverage / evaluated_questions:.3f}"
    )

    print(f"Evaluated Questions: {evaluated_questions}")
    print(f"Questions Without PDF Context: {no_context_questions}")
