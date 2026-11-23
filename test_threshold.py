
import json
from vectoredb import search

K = 5

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.65, 0.70]

with open("../evaluation/evaluation_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)


for threshold in THRESHOLDS:

    precisions_at_k = []
    precisions_retrieved = []
    recalls = []
    reciprocal_ranks = []

    total_retrieved = 0
    total_empty = 0

    print(f"\n{'=' * 50}")
    print(f"THRESHOLD: {threshold}")
    print(f"{'=' * 50}")

    for item in dataset:

        relevant_ids = set(item["relevant_chunk_ids"])

        if not relevant_ids:
            continue

        results = search(
            item["question"],
            k=K,
            threshold=threshold
        )

        retrieved_ids = [chunk["id"] for chunk in results]

        relevant_found = sum(
            1 for chunk_id in retrieved_ids
            if chunk_id in relevant_ids
        )

        precision_at_k = relevant_found / K

        precision_retrieved = (
            relevant_found / len(retrieved_ids)
            if retrieved_ids else 0
        )

        recall = relevant_found / len(relevant_ids)

        reciprocal_rank = 0

        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in relevant_ids:
                reciprocal_rank = 1 / rank
                break

        precisions_at_k.append(precision_at_k)
        precisions_retrieved.append(precision_retrieved)
        recalls.append(recall)
        reciprocal_ranks.append(reciprocal_rank)

        total_retrieved += len(retrieved_ids)

        if not retrieved_ids:
            total_empty += 1

    n = len(precisions_at_k)

    if n == 0:
        continue

    print(f"Precision@5: {sum(precisions_at_k) / n:.3f}")
    print(f"Precision@Retrieved: {sum(precisions_retrieved) / n:.3f}")
    print(f"Recall@5: {sum(recalls) / n:.3f}")
    print(f"MRR: {sum(reciprocal_ranks) / n:.3f}")
    print(f"Average Retrieved: {total_retrieved / n:.2f}")
    print(f"Empty Results: {total_empty}/{n}")
