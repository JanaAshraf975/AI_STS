
import json
from rag import retrieve_chunks

with open("../evaluation/evaluation_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

precisions = []
recalls = []
reciprocal_ranks = []
hop_correct = []

for item in dataset:
    question = item["question"]
    relevant_ids = set(item["relevant_chunk_ids"])

    if not relevant_ids:
        print(f"SKIPPED: {question} (no relevant IDs)")
        continue

    # Adaptive retrieval
    results, hop_type = retrieve_chunks(question, k=5)

    retrieved_ids = [chunk["id"] for chunk in results[:5]]

    # Count relevant chunks
    relevant_found = sum(
        1 for chunk_id in retrieved_ids
        if chunk_id in relevant_ids
    )

    # Precision and Recall
    precision = (
        relevant_found / len(retrieved_ids)
        if retrieved_ids else 0
    )

    recall = relevant_found / len(relevant_ids)

    # Reciprocal Rank
    reciprocal_rank = 0

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            reciprocal_rank = 1 / rank
            break

    precisions.append(precision)
    recalls.append(recall)
    reciprocal_ranks.append(reciprocal_rank)

    # Evaluate hop classification
    expected_hop = item.get("expected_hop_type")

    if expected_hop:
        is_correct = hop_type == expected_hop
        hop_correct.append(is_correct)

    print(f"\nQuestion: {question}")
    print(f"Expected Hop: {expected_hop}")
    print(f"Actual Hop: {hop_type}")
    print(f"Retrieved Chunks: {len(retrieved_ids)}")
    print(f"Relevant Found: {relevant_found}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"MRR: {reciprocal_rank:.2f}")

if precisions:
    print("\n===== ADAPTIVE RETRIEVAL RESULTS =====")

    print(
        f"Average Precision: "
        f"{sum(precisions) / len(precisions):.3f}"
    )

    print(
        f"Average Recall: "
        f"{sum(recalls) / len(recalls):.3f}"
    )

    print(
        f"MRR: "
        f"{sum(reciprocal_ranks) / len(reciprocal_ranks):.3f}"
    )

    if hop_correct:
        print(
            f"Hop Classification Accuracy: "
            f"{sum(hop_correct) / len(hop_correct):.3f}"
        )
