
import sys
from pdf_utils import chunk_folder
from vectoredb import index_chunks, search, count

PDF_FOLDER = "../pdfs"


def run_index():
    chunks = chunk_folder(PDF_FOLDER)
    if not chunks:
        print(f"No PDFs found in folder '{PDF_FOLDER}'")
        return
    print(f"{len(chunks)} chunks")
    total = index_chunks(chunks, reset=True)
    print(f"Indexed: {total}")


def run_search():
    n = count()
    print(f"Chunks in the database: {n}")
    if n == 0:
        print("The database is empty. Run: uv run python test_retrieval.py index")
        return

    while True:
        q = input("\nQuestion (or q to quit): ").strip()
        if q.lower() == "q":
            break
        if not q:
            continue
        for r in search(q, k=5):
            print(f"\n[{r['score']:.3f}] {r['source']} p.{r['page']}")
            print(r["text"][:300].replace("\n", " "), "...")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "index":
        run_index()
    else:
        run_search()
