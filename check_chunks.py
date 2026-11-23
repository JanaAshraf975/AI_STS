
from pdf_utils import chunk_folder
from embedding import count_tokens

PDF_FOLDER = "../pdfs"

chunks = chunk_folder(PDF_FOLDER)

texts = [chunk["text"] for chunk in chunks]

usage = count_tokens(texts)

print("\n========== TOKEN ANALYSIS ==========")
print("Total Chunks:", usage["number_of_texts"])
print("Total Tokens:", usage["total_tokens"])
print("Average Tokens per Chunk:", round(usage["average_tokens"], 2))
print("Min Tokens:", usage["min_tokens"])
print("Max Tokens:", usage["max_tokens"])
print("Model Max Length:", usage["model_max_length"])
print("Chunks Over Limit:", usage["chunks_over_limit"])

print("\nFirst 10 Chunks:")

for i, count in enumerate(usage["tokens_per_text"][:10]):
    print(f"Chunk {i + 1}: {count} tokens")
