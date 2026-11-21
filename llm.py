
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Model names live here so you can swap them without touching other files.
# Check openrouter.ai/models for current names and prices.
ROUTER_MODEL = "openai/gpt-4o-mini"
GENERATOR_MODEL = "openai/gpt-4o-mini"
JUDGE_MODEL = "openai/gpt-4o-mini"

# USD per 1 million tokens
INPUT_PRICE_PER_MILLION = 0.15
OUTPUT_PRICE_PER_MILLION = 0.60

_client = None


# ==============================
# Usage Tracking
# ==============================

_usage_totals = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "calls": 0,
}


def reset_usage_totals():
    """Reset usage counters before a new RAG request."""
    global _usage_totals

    _usage_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    }


def get_usage_totals():
    """Return total token usage and estimated cost."""
    totals = _usage_totals.copy()

    costs = calculate_token_cost(
        totals["prompt_tokens"],
        totals["completion_tokens"]
    )

    return {
        **totals,
        **costs
    }


# ==============================
# OpenRouter Client
# ==============================

def _get_client():
    global _client

    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing. Add it to your .env file."
            )

        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    return _client


# ==============================
# Token Cost Calculator
# ==============================

def calculate_token_cost(prompt_tokens, completion_tokens):
    """
    Calculate estimated LLM API cost in USD.

    Returns None for costs when token usage is unavailable.
    """

    input_cost = (
        prompt_tokens * INPUT_PRICE_PER_MILLION / 1_000_000
        if prompt_tokens is not None
        else None
    )

    output_cost = (
        completion_tokens * OUTPUT_PRICE_PER_MILLION / 1_000_000
        if completion_tokens is not None
        else None
    )

    total_cost = (
        input_cost + output_cost
        if input_cost is not None and output_cost is not None
        else None
    )

    return {
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
    }


# ==============================
# LLM Request
# ==============================

def ask_llm(
    prompt,
    model=GENERATOR_MODEL,
    system="Answer using ONLY the provided document context. "
        "Cite sources like [1], [2] after the claims they support. "
        "If the context does not contain the answer, reply exactly: "
        "\"I could not find this in the documents.\"",
    temperature=0,
    max_tokens=800,
    json_mode=False,
    retries=3,
    return_meta=False
):
    """
    Sends a prompt to OpenRouter and returns the answer text.

    If return_meta=True:
        Returns (text, meta).

    Every successful API call is tracked in usage totals,
    whether return_meta is True or False.
    """

    global _usage_totals

    messages = []

    if system:
        messages.append({
            "role": "system",
            "content": system
        })

    messages.append({
        "role": "user",
        "content": prompt
    })

    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if json_mode:
        kwargs["response_format"] = {
            "type": "json_object"
        }

    last_error = None

    for attempt in range(retries):
        try:
            start = time.time()

            resp = _get_client().chat.completions.create(**kwargs)

            latency = time.time() - start

            text = resp.choices[0].message.content or ""

            # ==============================
            # Extract Token Usage
            # ==============================

            usage = resp.usage

            prompt_tokens = getattr(
                usage,
                "prompt_tokens",
                None
            )

            completion_tokens = getattr(
                usage,
                "completion_tokens",
                None
            )

            total_tokens = getattr(
                usage,
                "total_tokens",
                None
            )

            if (
                total_tokens is None
                and prompt_tokens is not None
                and completion_tokens is not None
            ):
                total_tokens = prompt_tokens + completion_tokens

            # ==============================
            # Update Global Usage Totals
            # ==============================

            _usage_totals["calls"] += 1

            if prompt_tokens is not None:
                _usage_totals["prompt_tokens"] += prompt_tokens

            if completion_tokens is not None:
                _usage_totals["completion_tokens"] += completion_tokens

            if total_tokens is not None:
                _usage_totals["total_tokens"] += total_tokens

            # ==============================
            # Calculate Cost
            # ==============================

            costs = calculate_token_cost(
                prompt_tokens,
                completion_tokens
            )

            # Return text only when metadata is not requested.
            # Usage has already been recorded above.
            if not return_meta:
                return text

            # ==============================
            # Metadata
            # ==============================

            meta = {
                "model": model,
                "latency": round(latency, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                **costs
            }

            return text, meta

        except Exception as e:
            last_error = e

            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"LLM call failed after {retries} attempts: {last_error}"
    )


# ==============================
# Standalone Testing
# ==============================

if __name__ == "__main__":

    while True:
        question = input(
            "Enter your question (or 'q' to quit): "
        ).strip()

        if question.lower() == "q":
            break

        if not question:
            continue

        reset_usage_totals()

        answer, meta = ask_llm(
            question,
            return_meta=True
        )

        print("\nAnswer:")
        print(answer)

        print("\n--- TOKEN USAGE ---")
        print("Input Tokens:", meta["prompt_tokens"])
        print("Output Tokens:", meta["completion_tokens"])
        print("Total Tokens:", meta["total_tokens"])

        print("\n--- REQUEST COST ---")
        print("Input Cost: ${:.10f}".format(meta["input_cost_usd"]))
        print("Output Cost: ${:.10f}".format(meta["output_cost_usd"]))
        print("Total Cost: ${:.10f}".format(meta["total_cost_usd"]))

        totals = get_usage_totals()

        print("\n--- TOTAL USAGE ---")
        print("LLM Calls:", totals["calls"])
        print("Total Tokens:", totals["total_tokens"])
        print("Total Cost: ${:.10f}".format(totals["total_cost_usd"]))
