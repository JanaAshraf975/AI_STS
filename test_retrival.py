
import sys
from pdf_utils import chunk_folder
from vectoredb import index_chunks, search, count
from vectoredb import get_collection

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
        print("The database is empty. Run: uv run python test_retrival.py index")
        return

    while True:
        q = input("\nQuestion (or q to quit): ").strip()

        if q.lower() == "q":
            break

        if not q:
            continue

        results = search(q, k=5)

        for r in results:
            print(f"\n[{r['score']:.3f}] {r['source']} p.{r['page']}")
            print(r["text"][:300].replace("\n", " "), "...")


def show_chunks():
    collection = get_collection()

    chunk_ids = [
        "coralina_island_guide_p3_c1",
        "coralina_island_guide_p50_c0",
        "coralina_island_guide_p8_c0"
    ]

    results = collection.get(
        ids=chunk_ids,
        include=["documents", "metadatas"]
    )

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    if not results["ids"]:
        print("No matching chunks found.")
        return

    for chunk_id, document, metadata in zip(
        results["ids"],
        documents,
        metadatas
    ):
        print("\n" + "=" * 80)
        print(f"CHUNK ID: {chunk_id}")
        print(f"METADATA: {metadata}")
        print("-" * 80)
        print(document)


if __name__ == "__main__":

    if len(sys.argv) > 1:

        command = sys.argv[1].lower()

        if command == "index":
            run_index()

        elif command == "show_chunks":
            show_chunks()

        else:
            print("Unknown command.")
            print("Available commands: index, show_chunks")

    else:
        run_search()
