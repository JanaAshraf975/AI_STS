
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import json

# 1. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Load FAISS index
with open("faiss.pkl", "rb") as file:
    saved_bytes = pickle.load(file)

index = faiss.deserialize_index(
    np.frombuffer(saved_bytes, dtype="uint8")
)

# 3. Load original text chunks
with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

print("Ready! Ask questions about your PDF.")

# 4. Ask questions repeatedly
while True:
    query = input("\nYour question (or 'exit' to stop): ")

    if query.lower() == "exit":
        break

    # 5. Convert question into an embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # 6. Search for the 3 most similar chunks
    k = 3
    scores, indices = index.search(
        query_embedding,
        min(k, index.ntotal)
    )

    # 7. Display retrieved chunks
    print("\n--- Retrieved Chunks ---")

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):
        if idx == -1:
            continue

        print(f"\nResult {rank}")
        print(f"Similarity score: {score:.4f}")
        print(chunks[idx])

print("\nGoodbye!")
