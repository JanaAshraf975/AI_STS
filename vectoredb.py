
import chromadb
import logging
import time

from embedding import embed_documents, embed_question
from logger_config import log_event

DB_PATH = "./chroma_db"
COLLECTION = "docs"

_client = None


def elapsed_ms(start):
    return round((time.perf_counter() - start) * 1000, 2)


def _get_client():
    global _client

    if _client is None:
        start = time.perf_counter()

        try:
            _client = chromadb.PersistentClient(path=DB_PATH)

            log_event(
                logging.INFO,
                "vectoredb",
                "connection_created",
                "ChromaDB persistent client initialized",
                db_path=DB_PATH,
                latency_ms=elapsed_ms(start)
            )

        except Exception:
            log_event(
                logging.ERROR,
                "vectoredb",
                "connection_failed",
                "Failed to initialize ChromaDB client",
                db_path=DB_PATH,
                latency_ms=elapsed_ms(start)
            )
            raise

    return _client


def get_collection(reset=False):
    start = time.perf_counter()

    try:
        client = _get_client()

        if reset:
            try:
                client.delete_collection(COLLECTION)

                log_event(
                    logging.WARNING,
                    "vectoredb",
                    "collection_reset",
                    "Existing collection deleted",
                    collection=COLLECTION
                )

            except Exception as e:
                log_event(
                    logging.WARNING,
                    "vectoredb",
                    "collection_reset_skipped",
                    "Collection deletion was unsuccessful",
                    collection=COLLECTION,
                    error_type=type(e).__name__
                )

        collection = client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        log_event(
            logging.DEBUG,
            "vectoredb",
            "collection_ready",
            "ChromaDB collection is ready",
            collection=COLLECTION,
            reset=reset,
            latency_ms=elapsed_ms(start)
        )

        return collection

    except Exception as e:
        log_event(
            logging.ERROR,
            "vectoredb",
            "collection_failed",
            "Failed to access ChromaDB collection",
            collection=COLLECTION,
            error_type=type(e).__name__,
            latency_ms=elapsed_ms(start)
        )
        raise


def index_chunks(chunks, reset=False, batch=256):
    """chunks: list of {"id", "text", "source", "page"}"""

    start = time.perf_counter()

    try:
        collection = get_collection(reset=reset)

        log_event(
            logging.INFO,
            "vectoredb",
            "indexing_started",
            "Starting chunk indexing",
            chunks_count=len(chunks),
            batch_size=batch,
            reset=reset
        )

        for i in range(0, len(chunks), batch):
            batch_start = time.perf_counter()

            part = chunks[i:i + batch]
            texts = [c["text"] for c in part]

            collection.upsert(
                ids=[c["id"] for c in part],
                documents=texts,
                embeddings=embed_documents(texts),
                metadatas=[
                    {"source": c["source"], "page": c["page"]}
                    for c in part
                ],
            )

            log_event(
                logging.DEBUG,
                "vectoredb",
                "batch_indexed",
                "Chunk batch indexed successfully",
                batch_number=(i // batch) + 1,
                chunks_indexed=len(part),
                latency_ms=elapsed_ms(batch_start)
            )

        total = collection.count()

        log_event(
            logging.INFO,
            "vectoredb",
            "indexing_completed",
            "Chunk indexing completed",
            chunks_submitted=len(chunks),
            collection_count=total,
            latency_ms=elapsed_ms(start)
        )

        return total

    except Exception as e:
        log_event(
            logging.ERROR,
            "vectoredb",
            "indexing_failed",
            "Chunk indexing failed",
            error_type=type(e).__name__,
            latency_ms=elapsed_ms(start)
        )
        raise


def search(question, k=5, source=None):
    """source: optional, limits the search to one PDF"""

    start = time.perf_counter()

    log_event(
        logging.INFO,
        "vectoredb",
        "search_started",
        "Starting vector similarity search",
        top_k=k,
        source=source,
        question_length=len(question)
    )

    try:
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
                "score": 1 - res["distances"][0][i],
            })

        log_event(
            logging.INFO,
            "vectoredb",
            "search_completed",
            "Vector similarity search completed",
            top_k=k,
            source=source,
            results_count=len(results),
            similarity_scores=[
                round(r["score"], 4) for r in results
            ],
            chunk_ids=[r["id"] for r in results],
            latency_ms=elapsed_ms(start)
        )

        return results

    except Exception as e:
        log_event(
            logging.ERROR,
            "vectoredb",
            "search_failed",
            "Vector similarity search failed",
            top_k=k,
            source=source,
            error_type=type(e).__name__,
            latency_ms=elapsed_ms(start)
        )
        raise


def count():
    start = time.perf_counter()

    try:
        total = get_collection().count()

        log_event(
            logging.DEBUG,
            "vectoredb",
            "collection_count",
            "Collection chunk count retrieved",
            chunks_count=total,
            latency_ms=elapsed_ms(start)
        )

        return total

    except Exception as e:
        log_event(
            logging.ERROR,
            "vectoredb",
            "count_failed",
            "Failed to count collection chunks",
            error_type=type(e).__name__,
            latency_ms=elapsed_ms(start)
        )
        raise


def get_all_chunks(batch_size=1000, source=None):
    start = time.perf_counter()

    try:
        collection = get_collection()

        all_chunks = []
        offset = 0
        batches_read = 0

        log_event(
            logging.DEBUG,
            "vectoredb",
            "load_chunks_started",
            "Starting to load stored chunks",
            batch_size=batch_size,
            source=source
        )

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
            batches_read += 1

            log_event(
                logging.DEBUG,
                "vectoredb",
                "chunk_batch_loaded",
                "Loaded a batch of stored chunks",
                batch_number=batches_read,
                chunks_loaded=len(res["ids"]),
                total_loaded=len(all_chunks)
            )

        log_event(
            logging.INFO,
            "vectoredb",
            "load_chunks_completed",
            "Finished loading stored chunks",
            source=source,
            chunks_count=len(all_chunks),
            batches_read=batches_read,
            latency_ms=elapsed_ms(start)
        )

        return all_chunks

    except Exception as e:
        log_event(
            logging.ERROR,
            "vectoredb",
            "load_chunks_failed",
            "Failed to load stored chunks",
            source=source,
            error_type=type(e).__name__,
            latency_ms=elapsed_ms(start)
        )
        raise
