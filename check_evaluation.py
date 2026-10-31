
import json
from vectoredb import search

with open("../evaluation/evaluation_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

for item in dataset:
    print("\n" + "=" * 60)
    print("QUESTION:", item["question"])

    results = search(item["question"], k=5)

    for chunk in results:
        print(
            f'\nID: {chunk["id"]}'
            f'\nSource: {chunk["source"]}'
            f'\nPage: {chunk["page"]}'
            f'\nText: {chunk["text"][:250]}'
        )
