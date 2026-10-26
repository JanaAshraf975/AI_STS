import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


def load_prompt():
    with open("prompt/rag_prompt.txt", "r", encoding="utf-8") as f:
        return f.read()


PROMPT_TEMPLTE = load_prompt()


def rewrite_query(question, history=None):
   
    if not history:
        return question

    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    prompt = (
        "Given the conversation below and a follow-up question, rewrite the "
        "follow-up as a standalone question that is clear and self-contained "
        "for searching a document. Keep the same language. "
        "Return only the rewritten question.\n\n"
        f"Conversation:\n{convo}\n\n"
        f"Follow-up question: {question}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def ask_gemin(question, context, history=None):
    prompt = PROMPT_TEMPLTE.format(context=context, question=question)

    messages = []
    if history:
        messages.extend(history[-6:])  
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return response.choices[0].message.content