
import json
import time
from collections import defaultdict
from rag import answer

with open("../evaluation/evaluation_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

times = []
times_by_hop = defaultdict(list)

for item in dataset:
    print("\n" + "=" * 50)
    print("Question:", item["question"])

    start = time.perf_counter()

    try:
        result = answer(item["question"])
        elapsed = time.perf_counter() - start

        times.append(elapsed)

        hop_type = result.get("hop_type") or result["route"]
        times_by_hop[hop_type].append(elapsed)

        print(f'Route: {result["route"]}')
        print(f'Hop type: {hop_type}')
        print(f'Latency: {elapsed:.2f} seconds')

    except Exception as e:
        print("Error:", e)

if times:
    print("\n===== LATENCY RESULTS =====")
    print(f"Successful questions: {len(times)}")
    print(f"Average latency: {sum(times) / len(times):.2f} seconds")
    print(f"Fastest: {min(times):.2f} seconds")
    print(f"Slowest: {max(times):.2f} seconds")

    print("\n===== BY STRATEGY =====")
    for strategy, values in times_by_hop.items():
        avg = sum(values) / len(values)
        print(f"{strategy}: {avg:.2f} seconds ({len(values)} questions)")
