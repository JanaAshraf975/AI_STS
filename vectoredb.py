
import chromadb
import hashlib

from embedding import embed_documents, embed_question

DB_PATH = "./chroma_db"
COLLECTION = "docs"
SUMMARY_COLLECTION = "pdf_summaries"

_client = None


def _get_client():
    global _client

    if _client is None:
        _client = chromadb.PersistentClient(path=DB_PATH)

    return _client


def get_collection(collection_name=COLLECTION, reset=False):
    client = _get_client()

    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

def get_summary_collection():
    client = _get_client()

    return client.get_or_create_collection(
        name=SUMMARY_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks, reset=False, batch=256, collection_name=COLLECTION):
    collection = get_collection(
        collection_name=collection_name,
        reset=reset
    )

    for start in range(0, len(chunks), batch):
        batch_chunks = chunks[start:start + batch]

        ids = [chunk["id"] for chunk in batch_chunks]
        documents = [chunk["text"] for chunk in batch_chunks]
        metadatas = [
            {
                "source": chunk["source"],
                "page": chunk["page"]
            }
            for chunk in batch_chunks
        ]

        embeddings = embed_documents(documents)

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

    print(f"Indexed {len(chunks)} chunks in '{collection_name}'")

def save_summary(source, summary):
    """Save or update one final summary for a PDF."""

    if not summary or not summary.strip():
        return False

    collection = get_summary_collection()

    summary_id = "summary_" + hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()

    collection.upsert(
        ids=[summary_id],
        documents=[summary],
        embeddings=embed_documents([summary]),
        metadatas=[{"source": source}],
    )

    return True


def get_summary(source):
    """Retrieve the saved summary of one PDF."""

    collection = get_summary_collection()

    result = collection.get(
        where={"source": source},
        include=["documents", "metadatas"],
    )

    if not result["ids"]:
        return None

    return {
        "source": result["metadatas"][0]["source"],
        "text": result["documents"][0],
    }


def search(
    question,
    k=5,
    source=None,
    collection_name=COLLECTION,
    threshold=0.65
):
    collection = get_collection(collection_name=collection_name)

    query_embedding = embed_question(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"source": source} if source else None
    )

    output = []

    for i in range(len(results["ids"][0])):

        similarity = 1 - results["distances"][0][i]

        # Ignore chunks below the similarity threshold
        if similarity < threshold:
            continue

        output.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "score": similarity
        })

    return output


def count():
    return get_collection().count()


def get_all_chunks(batch_size=1000, source=None, collection_name=COLLECTION):
    collection = get_collection(collection_name=collection_name)

    all_chunks = []
    offset = 0

    while True:
        res = collection.get(
            limit=batch_size,
            offset=offset,
            where={"source": source} if source else None,
            include=["documents", "metadatas"]
        )

        if not res["ids"]:
            break

        for i in range(len(res["ids"])):
            metadata = res["metadatas"][i]

            all_chunks.append({
                "id": res["ids"][i],
                "text": res["documents"][i],
                "source": metadata["source"],
                "page": metadata["page"]
            })

        offset += len(res["ids"])

    return all_chunks
