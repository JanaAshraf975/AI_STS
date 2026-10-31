from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def load_model():
    """Load the embedding model once and reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_documents(texts, batch_size=32):
    """Convert document chunks into normalized embeddings (list of lists)."""
    model = load_model()
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()


def embed_question(question):
    """Convert a user query into a normalized embedding (list of floats)."""
    model = load_model()
    return model.encode(
        QUERY_PREFIX + question,
        normalize_embeddings=True,
    ).tolist()