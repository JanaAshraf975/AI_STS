
import os
import time
import logging
from dotenv import load_dotenv
from openai import OpenAI

from logger_config import log_event

load_dotenv()

# Model configuration
ROUTER_MODEL = "openai/gpt-4o-mini"
GENERATOR_MODEL = "openai/gpt-4o-mini"
JUDGE_MODEL = "openai/gpt-4o-mini"

_client = None


def _get_client():
    global _client

    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            log_event(
                logging.ERROR,
                "llm",
                "api_key_missing",
                "OpenRouter API key is missing"
            )
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing. Add it to your .env file."
            )

        try:
            _client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

            log_event(
                logging.INFO,
                "llm",
                "client_initialized",
                "OpenRouter client initialized"
            )

        except Exception as e:
            log_event(
                logging.ERROR,
                "llm",
                "client_initialization_failed",
                "Failed to initialize OpenRouter client",
                error_type=type(e).__name__
            )
            raise

    return _client


def ask_llm(
    prompt,
    model=GENERATOR_MODEL,
    system=None,
    temperature=0,
    max_tokens=1000,
    json_mode=False,
    retries=3,
    return_meta=False
):
    """
    Sends a prompt to OpenRouter and returns the answer text.
    If return_meta=True, returns (text, meta).
    """

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

    log_event(
        logging.INFO,
        "llm",
        "request_started",
        "Starting LLM API request",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
        prompt_length=len(prompt),
        retries=retries
    )

    last_error = None

    for attempt in range(retries):

        start = time.perf_counter()

        try:
            log_event(
                logging.DEBUG,
                "llm",
                "attempt_started",
                "Sending request to OpenRouter",
                model=model,
                attempt=attempt + 1,
                max_attempts=retries
            )

            resp = _get_client().chat.completions.create(**kwargs)

            latency = time.perf_counter() - start

            text = resp.choices[0].message.content or ""

            usage = resp.usage

            prompt_tokens = getattr(
                usage, "prompt_tokens", None
            )

            completion_tokens = getattr(
                usage, "completion_tokens", None
            )

            total_tokens = getattr(
                usage, "total_tokens", None
            )

            log_event(
                logging.INFO,
                "llm",
                "request_completed",
                "LLM API request completed successfully",
                model=model,
                attempt=attempt + 1,
                latency_ms=round(latency * 1000, 2),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                response_length=len(text)
            )

            if not return_meta:
                return text

            meta = {
                "model": model,
                "latency": round(latency, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }

            return text, meta

        except Exception as e:

            last_error = e

            log_event(
                logging.ERROR,
                "llm",
                "attempt_failed",
                "LLM API request attempt failed",
                model=model,
                attempt=attempt + 1,
                max_attempts=retries,
                error_type=type(e).__name__,
                latency_ms=round(
                    (time.perf_counter() - start) * 1000, 2
                )
            )

            if attempt < retries - 1:
                wait_seconds = 2 ** attempt

                log_event(
                    logging.WARNING,
                    "llm",
                    "retry_scheduled",
                    "Retrying LLM API request",
                    next_attempt=attempt + 2,
                    wait_seconds=wait_seconds
                )

                time.sleep(wait_seconds)

    log_event(
        logging.ERROR,
        "llm",
        "request_failed",
        "LLM API request failed after all attempts",
        model=model,
        attempts=retries,
        error_type=type(last_error).__name__
        if last_error else None
    )

    raise RuntimeError(
        f"LLM call failed after {retries} attempts: {last_error}"
    )


if __name__ == "__main__":

    while True:
        question = input(
            "Enter your question (or 'q' to quit): "
        ).strip()

        if question.lower() == "q":
            break

        if not question:
            continue

        answer, meta = ask_llm(
            question,
            return_meta=True
        )

        print("\nAnswer:")
        print(answer)

        print("\nMetadata:")
        print(meta)
