
import chromadb
import hashlib

from embedding import embed_documents, embed_question


DB_PATH = "./chroma_db"

COLLECTION = "docs"
SUMMARY_COLLECTION = "pdf_summaries"

CANDIDATE_MULTIPLIER = 10

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
        metadata={"hnsw:space": "cosine"}
    )


def index_chunks(
    chunks,
    reset=False,
    batch=256,
    collection_name=COLLECTION
):
    if not chunks:
        print("No chunks to index.")
        return

    collection = get_collection(
        collection_name=collection_name,
        reset=reset
    )

    total = len(chunks)

    for start in range(0, total, batch):

        batch_chunks = chunks[start:start + batch]

        ids = [
            chunk["id"]
            for chunk in batch_chunks
        ]

        documents = [
            chunk["text"]
            for chunk in batch_chunks
        ]

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

        print(
            f"Indexed {min(start + batch, total)}/{total}"
        )

    print(
        f"Successfully indexed {total} chunks "
        f"in '{collection_name}'"
    )


def save_summary(source, summary):

    if not summary or not summary.strip():
        return False

    collection = get_summary_collection()

    summary_id = "summary_" + hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()

    embeddings = embed_documents([summary])

    collection.upsert(
        ids=[summary_id],
        documents=[summary],
        embeddings=embeddings,
        metadatas=[{"source": source}]
    )

    return True


def get_summary(source):

    collection = get_summary_collection()

    result = collection.get(
        where={"source": source},
        include=["documents", "metadatas"]
    )

    if not result["ids"]:
        return None

    return {
        "source": result["metadatas"][0]["source"],
        "text": result["documents"][0]
    }


def search(
    question,
    k=3,
    threshold=0.65,
    source=None,
    collection_name=COLLECTION,
    candidate_multiplier=CANDIDATE_MULTIPLIER
):
    """
    Semantic search using embeddings and cosine similarity.

    1. Embed the question.
    2. Retrieve more candidates than k.
    3. Print raw Top 10 results before threshold.
    4. Convert cosine distance to similarity.
    5. Apply similarity threshold.
    6. Remove duplicate chunks.
    7. Sort by similarity score.
    8. Return top k results.
    """

    if not question or not question.strip() or k <= 0:
        return []

    collection = get_collection(
        collection_name=collection_name
    )

    total_count = collection.count()

    if total_count == 0:
        return []

    query_embedding = embed_question(question)

    candidate_k = min(
        max(k * candidate_multiplier, 20),
        total_count
    )

    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": candidate_k,
        "include": [
            "documents",
            "metadatas",
            "distances"
        ]
    }

    if source:
        query_args["where"] = {"source": source}

    results = collection.query(**query_args)

    if not results["ids"] or not results["ids"][0]:
        print("No raw retrieval results.")
        return []

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # DEBUG: Print raw results BEFORE threshold
    print("\n========== RAW RETRIEVAL TOP 10 ==========")
    print(f"Question: {question}")
    print(f"Candidates retrieved: {len(ids)}")
    print(f"Threshold: {threshold}")

    for i in range(min(10, len(ids))):

        distance = float(distances[i])
        similarity = 1.0 - distance
        metadata = metadatas[i] or {}

        print(
            f"Rank: {i + 1} | "
            f"ID: {ids[i]} | "
            f"Similarity: {similarity:.4f} | "
            f"Source: {metadata.get('source')} | "
            f"Page: {metadata.get('page')}"
        )

    print("===========================================\n")

    output = []
    seen_ids = set()

    for i in range(len(ids)):

        chunk_id = ids[i]

        if chunk_id in seen_ids:
            continue

        similarity = 1.0 - float(distances[i])

        if similarity < threshold:
            continue

        metadata = metadatas[i] or {}

        output.append({
            "id": chunk_id,
            "text": documents[i],
            "source": metadata.get("source"),
            "page": metadata.get("page"),
            "score": similarity
        })

        seen_ids.add(chunk_id)

    output.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    final_results = output[:k]

    print(
        f"Results after threshold: {len(output)} | "
        f"Final returned: {len(final_results)}"
    )

    return final_results


def count():
    return get_collection().count()


def get_all_chunks(
    batch_size=1000,
    source=None,
    collection_name=COLLECTION
):

    collection = get_collection(
        collection_name=collection_name
    )

    all_chunks = []
    offset = 0

    while True:

        get_args = {
            "limit": batch_size,
            "offset": offset,
            "include": [
                "documents",
                "metadatas"
            ]
        }

        if source:
            get_args["where"] = {"source": source}

        res = collection.get(**get_args)

        if not res["ids"]:
            break

        for i in range(len(res["ids"])):

            metadata = res["metadatas"][i] or {}

            all_chunks.append({
                "id": res["ids"][i],
                "text": res["documents"][i],
                "source": metadata.get("source"),
                "page": metadata.get("page")
            })

        offset += len(res["ids"])

    return all_chunks
