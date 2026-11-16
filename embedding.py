
import logging
import time

from sentence_transformers import SentenceTransformer
from logger_config import log_event


MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def elapsed_ms(start):
    return round((time.perf_counter() - start) * 1000, 2)


def load_model():
    """Load the embedding model once and reuse it."""

    global _model

    if _model is None:
        start = time.perf_counter()

        log_event(
            logging.INFO,
            "embedding",
            "model_loading_started",
            "Starting embedding model loading",
            model=MODEL_NAME
        )

        try:
            _model = SentenceTransformer(MODEL_NAME)

            log_event(
                logging.INFO,
                "embedding",
                "model_loaded",
                "Embedding model loaded successfully",
                model=MODEL_NAME,
                embedding_dimensions=_model.get_sentence_embedding_dimension(),
                latency_ms=elapsed_ms(start)
            )

        except Exception as e:
            log_event(
                logging.ERROR,
                "embedding",
                "model_loading_failed",
                "Failed to load embedding model",
                model=MODEL_NAME,
                error_type=type(e).__name__,
                latency_ms=elapsed_ms(start)
            )
            raise

    else:
        log_event(
            logging.DEBUG,
            "embedding",
            "model_reused",
            "Reusing previously loaded embedding model",
            model=MODEL_NAME
        )

    return _model


def embed_documents(texts, batch_size=32):
    """Convert document chunks into normalized embeddings."""

    start = time.perf_counter()

    log_event(
        logging.INFO,
        "embedding",
        "document_embedding_started",
        "Starting document embedding generation",
        model=MODEL_NAME,
        texts_count=len(texts),
        batch_size=batch_size,
        normalized=True
    )

    try:
        model = load_model()

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).tolist()

        dimensions = len(embeddings[0]) if embeddings else 0

        log_event(
            logging.INFO,
            "embedding",
            "document_embedding_completed",
            "Document embeddings generated successfully",
            model=MODEL_NAME,
            texts_count=len(texts),
            embeddings_count=len(embeddings),
            embedding_dimensions=dimensions,
            batch_size=batch_size,
            latency_ms=elapsed_ms(start)
        )

        return embeddings

    except Exception as e:
        log_event(
            logging.ERROR,
            "embedding",
            "document_embedding_failed",
            "Document embedding generation failed",
            model=MODEL_NAME,
            texts_count=len(texts),
            error_type=type(e).__name__,
            latency_ms=elapsed_ms(start)
        )
        raise


def embed_question(question):
    """Convert a user query into a normalized embedding."""

    start = time.perf_counter()

    log_event(
        logging.INFO,
        "embedding",
        "query_embedding_started",
        "Starting query embedding generation",
        model=MODEL_NAME,
        question_length=len(question),
        normalized=True
    )

    try:
        model = load_model()

        embedding = model.encode(
            QUERY_PREFIX + question,
            normalize_embeddings=True,
        ).tolist()

        log_event(
            logging.INFO,
            "embedding",
            "query_embedding_completed",
            "Query embedding generated successfully",
            model=MODEL_NAME,
            embedding_dimensions=len(embedding),
            latency_ms=elapsed_ms(start)
        )

        return embedding

    except Exception as e:
        log_event(
            logging.ERROR,
            "embedding",
            "query_embedding_failed",
            "Query embedding generation failed",
            model=MODEL_NAME,
            error_type=type(e).__name__,
            latency_ms=elapsed_ms(start)
        )
        raise
