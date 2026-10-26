
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import json
import pickle
import faiss
import numpy as np

# ==================================
# 1. Read PDF
# ==================================

pdf_path = "Machine_Learning_and_Deep_Learning.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text() or ""
    text += page_text + "\n"

if not text.strip():
    raise ValueError("No text found in PDF. It may be a scanned PDF.")

print("PDF text extracted successfully!")
print("Total characters:", len(text))


# ==================================
# 2. Split text into chunks
# ==================================

chunk_size = 500
overlap = 100

chunks = [
    text[i:i + chunk_size]
    for i in range(0, len(text), chunk_size - overlap)
    if text[i:i + chunk_size].strip()
]

if not chunks:
    raise ValueError("No text chunks were created.")

print("Number of chunks:", len(chunks))


# ==================================
# 3. Load embedding model
# ==================================

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully!")


# ==================================
# 4. Create embeddings
# ==================================

embeddings = model.encode(
    chunks,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")

print("Embeddings shape:", embeddings.shape)


# ==================================
# 5. Save chunks only in JSON
# ==================================

with open("embeddings.json", "w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=2)

print("Chunks saved to embeddings.json!")

# Save chunks in a separate file too
with open("chunks.json", "w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=2)

print("Text chunks saved to chunks.json!")


# ==================================
# 6. Create FAISS index
# ==================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print("FAISS index created successfully!")


# ==================================
# 7. Save FAISS index in faiss.pkl
# ==================================

index_bytes = faiss.serialize_index(index).tobytes()

with open("faiss.pkl", "wb") as file:
    pickle.dump(index_bytes, file)

print("FAISS index saved to faiss.pkl!")


# ==================================
# 8. Verify faiss.pkl
# ==================================

with open("faiss.pkl", "rb") as file:
    saved_bytes = pickle.load(file)

loaded_index = faiss.deserialize_index(
    np.frombuffer(saved_bytes, dtype="uint8")
)

print("\n--- FAISS Verification ---")
print("Number of vectors:", loaded_index.ntotal)
print("Vector dimension:", loaded_index.d)

# Display the first vector
first_vector = loaded_index.reconstruct(0)

print("First vector:", first_vector)
print("First 5 values:", first_vector[:5])

print("\nAll steps completed successfully!")
