from vectoredb import get_collection

collection = get_collection()

data = collection.get(include=["metadatas"])

pdf_names = set()

for metadata in data["metadatas"]:
    source = metadata.get("source")

    if source:
        pdf_names.add(source)

print("TOTAL PDFs:", len(pdf_names))
print("=" * 60)

for pdf in sorted(pdf_names):
    print(pdf)