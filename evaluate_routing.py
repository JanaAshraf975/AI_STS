
from rag import classify_retrieval

questions = [
    {
        "question": "How much does the DoorGuard Sensor cost?",
        "expected": "single-hop"
    },
    {
        "question": "Which Wi-Fi network settings are supported by AsterHome devices?",
        "expected": "single-hop"
    },
    {
        "question": "What is required for a warranty claim?",
        "expected": "single-hop"
    },
    {
        "question": "Compare the DoorGuard Sensor and ViewCam Indoor prices.",
        "expected": "multi-hop"
    },
    {
        "question": "What should a customer do if they have no invoice, and what documents are required for a warranty claim?",
        "expected": "multi-hop"
    }
]

correct = 0

for item in questions:
    predicted = classify_retrieval(item["question"])
    is_correct = predicted == item["expected"]
    correct += is_correct

    print(f'\nQuestion: {item["question"]}')
    print(f'Expected: {item["expected"]}')
    print(f'Predicted: {predicted}')
    print(f'Correct: {is_correct}')

print(f'\nRouting Accuracy: {correct / len(questions):.2%}')
