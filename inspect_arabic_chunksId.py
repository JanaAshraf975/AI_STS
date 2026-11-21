
from vectoredb import get_all_chunks

# Choose the experiment
collection_name = "solar_1000_200"

# Get all chunks from the Arabic PDF
chunks = get_all_chunks(
    source="solar_energy_arabic.pdf",
    collection_name=collection_name
)

print("Collection:", collection_name)
print("Total Arabic Chunks:", len(chunks))

for chunk in chunks:
    print("\n" + "=" * 60)
    print("ID:", chunk["id"])
    print("Page:", chunk["page"])
    print("Text:")
    print(chunk["text"])
