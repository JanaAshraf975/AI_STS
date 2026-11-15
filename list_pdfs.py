
from vectoredb import get_collection

collection = get_collection()

data = collection.get(include=["documents", "metadatas"])

pdf_chunks = {}

for chunk_id, text, metadata in zip(
    data["ids"],
    data["documents"],
    data["metadatas"]
):
    source = metadata["source"]

    if source not in pdf_chunks:
        pdf_chunks[source] = []

    pdf_chunks[source].append({
        "id": chunk_id,
        "page": metadata["page"],
        "text": text
    })
    


source = "03_AsterHome_Support_Policies.pdf"

chunks = pdf_chunks.get(source, [])

print("=" * 60)
print("PDF:", source)
print("TOTAL CHUNKS:", len(chunks))

for chunk in chunks:
    print("\n" + "-" * 60)
    print("CHUNK ID:", chunk["id"])
    print("PAGE:", chunk["page"])
    print("TEXT:", chunk["text"])
