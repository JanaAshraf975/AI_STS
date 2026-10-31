
import json
from vectoredb import search

K = 5

with open("../evaluation/evaluation_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

precisions = []
recalls = []
reciprocal_ranks = []

for item in dataset:
    relevant_ids = set(item["relevant_chunk_ids"])

    if not relevant_ids:
        print(f'SKIPPED: {item["question"]} (no relevant IDs)')
        continue

    results = search(item["question"], k=K)
    retrieved_ids = [chunk["id"] for chunk in results]

    relevant_found = sum(
        1 for chunk_id in retrieved_ids if chunk_id in relevant_ids
    )

    precision = relevant_found / K
    recall = relevant_found / len(relevant_ids)

    reciprocal_rank = 0
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            reciprocal_rank = 1 / rank
            break

    precisions.append(precision)
    recalls.append(recall)
    reciprocal_ranks.append(reciprocal_rank)

    print(f'\nQuestion: {item["question"]}')
    print(f'Precision@5: {precision:.2f}')
    print(f'Recall@5: {recall:.2f}')
    print(f'MRR contribution: {reciprocal_rank:.2f}')

if precisions:
    print("\n===== FINAL RESULTS =====")
    print(f'Average Precision@5: {sum(precisions) / len(precisions):.3f}')
    print(f'Average Recall@5: {sum(recalls) / len(recalls):.3f}')
    print(f'MRR: {sum(reciprocal_ranks) / len(reciprocal_ranks):.3f}')
