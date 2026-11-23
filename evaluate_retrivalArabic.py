
import json
from vectoredb import search

K = 5

with open("../evaluation/evaluation_dataset_longArabic.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

precisions = []
recalls = []
reciprocal_ranks = []

for item in dataset:
    relevant_ids = set(item["relevant_chunk_ids"])

    if not relevant_ids:
        print(f'SKIPPED: {item["question"]} (no relevant IDs)')
        continue

    # Retrieve top K chunks
    results = search(item["question"], k=K)
    retrieved_ids = [chunk["id"] for chunk in results]

    # Debug only the first question
    if item["question"] == "ما الفرق بين نماذج الترانسفورمر ذات المشفر فقط والمفكك فقط؟":

        print("\n" + "=" * 70)
        print("RETRIEVAL DEBUG")
        print("Question:", item["question"])

        print("\nExpected Relevant IDs:")
        for chunk_id in relevant_ids:
            print(chunk_id)

        print("\nRetrieved Top 5:")
        for rank, chunk in enumerate(results, start=1):
            print(f"\nRank {rank}")
            print("ID:", chunk["id"])
            print("Text:", chunk.get("text", "Text field not found")[:500])

        print("=" * 70)

    # Calculate metrics
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


# Final results
if precisions:
    print("\n===== FINAL RESULTS =====")

    print(
        f'Average Precision@5: '
        f'{sum(precisions) / len(precisions):.3f}'
    )

    print(
        f'Average Recall@5: '
        f'{sum(recalls) / len(recalls):.3f}'
    )

    print(
        f'MRR: '
        f'{sum(reciprocal_ranks) / len(reciprocal_ranks):.3f}'
    )
