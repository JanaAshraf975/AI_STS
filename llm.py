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

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to your .env file.")
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _client


def ask_llm(prompt, model=GENERATOR_MODEL, system=None, temperature=0,
            max_tokens=1000, json_mode=False, retries=3, return_meta=False):
    """
    Sends a prompt to OpenRouter and returns the answer text.
    If return_meta=True, returns (text, meta) where meta has latency and token usage.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_error = None
    for attempt in range(retries):
        try:
            start = time.time()
            resp = _get_client().chat.completions.create(**kwargs)
            latency = time.time() - start

            text = resp.choices[0].message.content or ""
            if not return_meta:
                return text

            usage = resp.usage
            meta = {
                "model": model,
                "latency": round(latency, 3),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            }
            return text, meta
        except Exception as e:
            last_error = e
            time.sleep(2 ** attempt)  # wait 1s, 2s, 4s between retries

    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_error}")



if __name__ == "__main__":
    while True:
        question = input("Enter your question (or 'q' to quit): ").strip()

        if question.lower() == "q":
            break

        if not question:
            continue

        answer, meta = ask_llm(question, return_meta=True)

        print("\nAnswer:")
        print(answer)
        print("\nMetadata:")
        print(meta)
