import chromadb
from embedding import embed_documents, embed_question

DB_PATH = "./chroma_db"
COLLECTION = "docs"

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=DB_PATH)
    return _client


def get_collection(reset=False):
    client = _get_client()
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks, reset=False, batch=256):
    """chunks: list of {"id", "text", "source", "page"}"""
    collection = get_collection(reset=reset)
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        texts = [c["text"] for c in part]
        collection.upsert(
            ids=[c["id"] for c in part],
            documents=texts,
            embeddings=embed_documents(texts),
            metadatas=[{"source": c["source"], "page": c["page"]} for c in part],
        )
    return collection.count()


def search(question, k=5, source=None):
    """source: optional, limits the search to one PDF"""
    collection = get_collection()
    res = collection.query(
        query_embeddings=[embed_question(question)],
        n_results=k,
        where={"source": source} if source else None,
    )
    results = []
    for i in range(len(res["ids"][0])):
        results.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "source": res["metadatas"][0][i]["source"],
            "page": res["metadatas"][0][i]["page"],
            "score": 1 - res["distances"][0][i],  # cosine similarity
        })
    return results


def count():
    return get_collection().count()