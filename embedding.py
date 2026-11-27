
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
import tiktoken

from openai import OpenAI
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# OpenRouter embedding model
MODEL_NAME = "qwen/qwen3-embedding-8b"

# Tokenizer settings
# Used for approximate token statistics.
# Qwen uses its own tokenizer, so tiktoken counts are estimates.
TOKENIZER_MODEL = "cl100k_base"
MAX_INPUT_TOKENS = 32768

BATCH_SIZE = 100

# Qwen3 retrieval instruction
QUERY_INSTRUCTION = (
    "Instruct: Given a question, retrieve relevant passages "
    "that answer the question.\nQuery: "
)

_model = None
_tokenizer = None


def load_model():
    """Initialize OpenRouter client once and reuse it."""

    global _model

    if _model is None:
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment variables."
            )

        _model = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

    return _model


def load_tokenizer():
    """Load tokenizer once and reuse it."""

    global _tokenizer

    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding(TOKENIZER_MODEL)

    return _tokenizer


def count_tokens(texts):
    """
    Count approximate tokens for a list of texts.

    Note:
    tiktoken is not Qwen's native tokenizer.
    Counts are useful for statistics, not exact limit validation.
    """

    tokenizer = load_tokenizer()

    tokens_per_text = [
        len(tokenizer.encode(text, disallowed_special=()))
        for text in texts
    ]

    total_tokens = sum(tokens_per_text)
    number_of_texts = len(texts)

    if number_of_texts == 0:
        return {
            "total_tokens": 0,
            "tokens_per_text": [],
            "number_of_texts": 0,
            "average_tokens": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "model_max_length": MAX_INPUT_TOKENS,
            "chunks_over_limit": []
        }

    chunks_over_limit = [
        i for i, count in enumerate(tokens_per_text)
        if count > MAX_INPUT_TOKENS
    ]

    return {
        "total_tokens": total_tokens,
        "tokens_per_text": tokens_per_text,
        "number_of_texts": number_of_texts,
        "average_tokens": total_tokens / number_of_texts,
        "min_tokens": min(tokens_per_text),
        "max_tokens": max(tokens_per_text),
        "model_max_length": MAX_INPUT_TOKENS,
        "chunks_over_limit": chunks_over_limit
    }


def embed_documents(texts, batch_size=BATCH_SIZE):
    """
    Generate document embeddings using Qwen3 through OpenRouter.

    Documents are processed in batches.
    The original order of embeddings is preserved.
    """

    if not texts:
        return []

    client = load_model()

    # Count tokens before embedding
    usage = count_tokens(texts)

    print(f"Chunks: {usage['number_of_texts']}")
    print(f"Total approximate tokens: {usage['total_tokens']}")
    print(
        f"Average tokens per chunk: "
        f"{usage['average_tokens']:.2f}"
    )
    print(f"Maximum tokens in a chunk: {usage['max_tokens']}")
    print(f"Model max length: {usage['model_max_length']}")
    print(
        f"Chunks over approximate model limit: "
        f"{len(usage['chunks_over_limit'])}"
    )

    if usage["chunks_over_limit"]:
        raise ValueError(
            f"Some chunks exceed the configured token limit. "
            f"Chunk indexes: {usage['chunks_over_limit'][:10]}"
        )

    all_embeddings = []

    # Process documents in batches
    for start in range(0, len(texts), batch_size):

        end = min(start + batch_size, len(texts))
        batch = texts[start:end]

        response = client.embeddings.create(
            model=MODEL_NAME,
            input=batch,
            encoding_format="float"
        )

        # Preserve original order
        batch_embeddings = sorted(
            response.data,
            key=lambda item: item.index
        )

        all_embeddings.extend([
            item.embedding for item in batch_embeddings
        ])

        print(f"Embedded {end}/{len(texts)}")

    return all_embeddings


def embed_question(question):
    """
    Generate query embedding using Qwen3.

    Adds a retrieval instruction to the question.
    """

    client = load_model()

    formatted_question = QUERY_INSTRUCTION + question

    # Count approximate query tokens
    usage = count_tokens([formatted_question])

    query_tokens = usage["tokens_per_text"][0]

    if query_tokens > MAX_INPUT_TOKENS:
        raise ValueError(
            f"Question exceeds the configured token limit. "
            f"Approximate tokens: {query_tokens}"
        )

    response = client.embeddings.create(
        model=MODEL_NAME,
        input=formatted_question,
        encoding_format="float"
    )

    return response.data[0].embedding
