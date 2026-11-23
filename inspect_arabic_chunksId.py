from pathlib import Path
from vectoredb import get_all_chunks

chunks = get_all_chunks(
    source="Building the Future Egypt’s Political and Social Achievements in 2024 and the Road to 2025.pdf"
)

output_path = Path("all_chunks.txt")

with output_path.open("w", encoding="utf-8") as file:

    file.write(f"Total Chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks, start=1):

        file.write("\n" + "=" * 60 + "\n")
        file.write(f"Chunk Number: {i}\n")
        file.write(f"ID: {chunk['id']}\n")
        file.write(f"Page: {chunk['page']}\n")
        file.write("Text:\n")
        file.write(chunk["text"])
        file.write("\n")

print(f"Successfully saved {len(chunks)} chunks!")
print(f"File location: {output_path.resolve()}")