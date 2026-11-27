
import json
from vectoredb import get_all_chunks

# Get all chunks from ChromaDB
chunks = get_all_chunks()

# Export chunks to JSON
output_path = "../evaluation/all_chunks.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"Exported {len(chunks)} chunks successfully!")
print(f"Saved to: {output_path}")
