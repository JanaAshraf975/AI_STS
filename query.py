
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import json

# ==================================
# 1. Load embedding model
# ==================================

model = SentenceTransformer("all-MiniLM-L6-v2")

# ==================================
# 2. Load FAISS index
# ==================================

with open("faiss.pkl", "rb") as file:
    saved_bytes = pickle.load(file)

index = faiss.deserialize_index(
    np.frombuffer(saved_bytes, dtype="uint8")
)

# ==================================
# 3. Load text chunks
# ==================================

with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

print("FAISS index and chunks loaded successfully!")

# ==================================
# 4. Enter your query
# ==================================

query = input("Enter your question: ")

# ==================================
# 5. Convert query to embedding
# ==================================

query_embedding = model.encode(
    [query],
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")

print("Query embedding shape:", query_embedding.shape)

# ==================================
# 6. Retrieve similar chunks
# ==================================

k = 3  # Number of chunks to retrieve

scores, indices = index.search(query_embedding, min(k, index.ntotal))

# ==================================
# 7. Display results
# ==================================

print("\n--- Retrieved Chunks ---")

for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
    if idx == -1:
        continue

    print(f"\nResult {rank}")
    print(f"Similarity score: {score:.4f}")
    print(f"Chunk index: {idx}")
    print(chunks[idx])

print("\nRetrieval completed!")



import ollama

# Collect the retrieved chunks
retrieved_chunks = [
    chunks[idx]
    for idx in indices[0]
    if idx != -1
]

context = "\n\n".join(retrieved_chunks)

# Build a prompt for the LLM
prompt = f"""
You are a helpful AI tutor.

Answer the user's question using the context from the PDF.
Explain the answer in simple English.
Customize your explanation to the question.
You may use examples to make the answer easier to understand.
Do not copy the context word for word.
If the context does not contain enough information, say so.

Context:
{context}

Question:
{query}

Answer:
"""

# Generate a customized answer
response = ollama.generate(
    model="gemma2:2b",
    prompt=prompt
)

print("\n--- Customized Answer ---")
print(response["response"])

