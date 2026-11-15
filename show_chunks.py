from vectoredb import search

questions = [
    "How much does the SecurePlug cost?",
    "What is the maximum power rating of SecurePlug?",
    "Does SecurePlug require a hub?",
    "Can SecurePlug be used with a 2,500 W heater?",
    "What features does SecurePlug provide?",
    "What video resolution does ViewCam Indoor support?",
    "Does ViewCam Indoor support two-way audio and night vision?",
    "Does ViewCam Indoor require a subscription for live view and motion alerts?",
    "What is required to access cloud event history on ViewCam Indoor?",
    "What is the maximum microSD card capacity supported by ViewCam Indoor?",
    "What wireless technology does ClimateSense use?",
    "What is the expected battery life of ClimateSense?"
]

for question in questions:
    print("\n" + "=" * 70)
    print("QUESTION:", question)

    results = search(question, k=5)

    for chunk in results:
        print(
            f'\nID: {chunk["id"]}'
            f'\nSource: {chunk["source"]}'
            f'\nPage: {chunk["page"]}'
            f'\nText: {chunk["text"]}'
        )