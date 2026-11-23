
# import os

# # Store Hugging Face models on D instead of C
# os.environ["HF_HOME"] = "D:/huggingface"

# from sentence_transformers import SentenceTransformer

# MODEL_NAME = "BAAI/bge-m3"
# QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# _model = None


# def load_model():
#     """Load the embedding model once and reuse it."""
#     global _model
#     if _model is None:
#         _model = SentenceTransformer(MODEL_NAME)
#     return _model


# def embed_documents(texts, batch_size=32):
#     """Convert document chunks into normalized embeddings (list of lists)."""
#     model = load_model()
#     return model.encode(
#         texts,
#         batch_size=batch_size,
#         normalize_embeddings=True,
#         show_progress_bar=True,
#     ).tolist()


# def embed_question(question):
#     """Convert a user query into a normalized embedding (list of floats)."""
#     model = load_model()
#     return model.encode(
#         QUERY_PREFIX + question,
#         normalize_embeddings=True,
#     ).tolist()




import os

# Store Hugging Face models on D instead of C
os.environ["HF_HOME"] = "D:/huggingface"

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def load_model():
    """Load the embedding model once and reuse it."""
    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def count_tokens(texts):
    """Count the actual tokens in each text without truncation."""

    model = load_model()
    tokenizer = model.tokenizer

    if isinstance(texts, str):
        texts = [texts]

    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )

    counts = [len(ids) for ids in encoded["input_ids"]]

    max_length = model.max_seq_length

    return {
        "total_tokens": sum(counts),
        "tokens_per_text": counts,
        "number_of_texts": len(texts),
        "average_tokens": (
            sum(counts) / len(counts) if counts else 0
        ),
        "min_tokens": min(counts) if counts else 0,
        "max_tokens": max(counts) if counts else 0,
        "model_max_length": max_length,
        "chunks_over_limit": sum(
            count > max_length for count in counts
        ),
    }


def embed_documents(texts, batch_size=32):
    """Convert document chunks into normalized embeddings."""

    model = load_model()

    usage = count_tokens(texts)

    print(f"Chunks: {usage['number_of_texts']}")
    print(f"Total embedding tokens: {usage['total_tokens']}")
    print(f"Average tokens per chunk: {usage['average_tokens']:.2f}")
    print(f"Maximum tokens in a chunk: {usage['max_tokens']}")
    print(f"Model max length: {usage['model_max_length']}")
    print(f"Chunks over model limit: {usage['chunks_over_limit']}")

    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()


def embed_question(question):
    """Convert a user query into a normalized embedding."""

    model = load_model()

    text = QUERY_PREFIX + question

    usage = count_tokens(text)

    print(f"Query embedding tokens: {usage['total_tokens']}")

    return model.encode(
        text,
        normalize_embeddings=True,
    ).tolist()
