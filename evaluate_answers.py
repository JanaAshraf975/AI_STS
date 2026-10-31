
import json
from rag import answer

with open("../evaluation/evaluation_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

for item in dataset:
    print("\n" + "=" * 60)
    print("Question:", item["question"])
    print("Expected:", item["expected_answer"])

    result = answer(item["question"])

    print("Generated:", result["answer"])
    print("Route:", result["route"])
    print("Hop type:", result["hop_type"])
    print("Retried:", result["retried"])
