
from vectoredb import get_collection

collection = get_collection()
data = collection.get(include=["documents", "metadatas"])

shown = set()

for doc, meta in zip(data["documents"], data["metadatas"]):
    source = meta["source"]

    if source not in shown:
        shown.add(source)
        print("\n" + "=" * 60)
        print("PDF:", source)
        print("PAGE:", meta["page"])
        print("CHUNK:", doc)
        
        if len(shown) == 3:
            break
