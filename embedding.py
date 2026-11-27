
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

from openai import OpenAI
from dotenv import load_dotenv
from transformers import AutoTokenizer


# Load environment variables
load_dotenv()


# OpenRouter embedding model
MODEL_NAME = "intfloat/multilingual-e5-large"

# Hugging Face tokenizer
TOKENIZER_MODEL = "intfloat/multilingual-e5-large"

# Model configuration
MAX_INPUT_TOKENS = 512
BATCH_SIZE = 100

# E5 prefixes
DOCUMENT_PREFIX = "passage: "
QUERY_PREFIX = "query: "

_model = None
_tokenizer = None


def load_model():
    """Initialize OpenRouter client once and reuse it."""

    global _model

    if _model is None:

        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. Check your .env file."
            )

        _model = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

    return _model


def load_tokenizer():
    """Load the original E5 tokenizer for accurate token counting."""

    global _tokenizer

    if _tokenizer is None:

        _tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_MODEL
        )

    return _tokenizer


def count_tokens(texts):
    """Count tokens using the original E5 tokenizer."""

    tokenizer = load_tokenizer()

    if isinstance(texts, str):
        texts = [texts]

    counts = [
        len(
            tokenizer.encode(
                text,
                add_special_tokens=True
            )
        )
        for text in texts
    ]

    return {
        "total_tokens": sum(counts),

        "tokens_per_text": counts,

        "number_of_texts": len(counts),

        "average_tokens": (
            sum(counts) / len(counts)
            if counts else 0
        ),

        "min_tokens": min(counts) if counts else 0,

        "max_tokens": max(counts) if counts else 0,

        "model_max_length": MAX_INPUT_TOKENS,

        "chunks_over_limit": sum(
            count > MAX_INPUT_TOKENS
            for count in counts
        ),
    }


def embed_documents(texts, batch_size=BATCH_SIZE):
    """
    Convert document chunks into Multilingual-E5 embeddings.

    E5 requires the passage: prefix for documents.
    """

    if not texts:
        return []

    if batch_size <= 0:
        raise ValueError(
            "Batch size must be greater than zero."
        )

    client = load_model()

    # Add E5 document prefix
    prepared_texts = [
        DOCUMENT_PREFIX + text
        for text in texts
    ]

    usage = count_tokens(prepared_texts)

    print(f"Chunks: {usage['number_of_texts']}")

    print(
        f"Total embedding tokens: "
        f"{usage['total_tokens']}"
    )

    print(
        f"Average tokens per chunk: "
        f"{usage['average_tokens']:.2f}"
    )

    print(
        f"Maximum tokens in a chunk: "
        f"{usage['max_tokens']}"
    )

    print(
        f"Model max length: "
        f"{usage['model_max_length']}"
    )

    print(
        f"Chunks over model limit: "
        f"{usage['chunks_over_limit']}"
    )

    if usage["chunks_over_limit"] > 0:

        raise ValueError(
            "Some chunks exceed the 512-token limit. "
            "Reduce chunk size before indexing."
        )

    all_embeddings = []

    for start in range(0, len(prepared_texts), batch_size):

        batch = prepared_texts[start:start + batch_size]

        response = client.embeddings.create(
            model=MODEL_NAME,
            input=batch,
            encoding_format="float"
        )

        # Preserve original input order
        batch_embeddings = [
            item.embedding
            for item in sorted(
                response.data,
                key=lambda item: item.index
            )
        ]

        all_embeddings.extend(batch_embeddings)

        print(
            f"Embedded "
            f"{min(start + batch_size, len(prepared_texts))}"
            f"/{len(prepared_texts)}"
        )

    return all_embeddings


def embed_question(question):
    """
    Convert a user query into a Multilingual-E5 embedding.

    E5 requires the query: prefix for questions.
    """

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    client = load_model()

    # Add E5 query prefix
    prepared_question = QUERY_PREFIX + question

    usage = count_tokens(prepared_question)

    print(
        f"Query embedding tokens: "
        f"{usage['total_tokens']}"
    )

    if usage["chunks_over_limit"] > 0:

        raise ValueError(
            "Question exceeds the 512-token limit."
        )

    response = client.embeddings.create(
        model=MODEL_NAME,
        input=prepared_question,
        encoding_format="float"
    )

    return response.data[0].embedding
