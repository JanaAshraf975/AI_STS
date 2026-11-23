
import os
import json
import re
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
from openai import OpenAI

from vectoredb import get_all_chunks


# =========================
# CONFIG
# =========================

MODEL = "openai/gpt-4o-mini"

GROUP_SIZE = 3
STRIDE = 2

OUTPUT_FILE = "evaluation2_dataset.json"

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is missing from .env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


# =========================
# HELPERS
# =========================

def natural_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def group_chunks(chunks):
    grouped_by_source = defaultdict(list)

    for chunk in chunks:
        grouped_by_source[chunk["source"]].append(chunk)

    groups = []

    for source, source_chunks in grouped_by_source.items():

        source_chunks.sort(
            key=lambda x: (
                int(x["page"]),
                natural_key(x["id"])
            )
        )

        for start in range(
            0,
            len(source_chunks) - GROUP_SIZE + 1,
            STRIDE
        ):
            group = source_chunks[start:start + GROUP_SIZE]
            groups.append(group)

    return groups


# =========================
# LLM GENERATION
# =========================

def generate_question(group):

    chunks_text = "\n\n".join(
        f"""
CHUNK ID: {chunk['id']}
PAGE: {chunk['page']}
CONTENT:
{chunk['text']}
"""
        for chunk in group
    )

    allowed_ids = [chunk["id"] for chunk in group]

    prompt = f"""
You are creating a high-quality Ground Truth dataset
for evaluating a Retrieval-Augmented Generation system.

Use ONLY the information in the supplied chunks.

Generate ONE natural, specific question and its correct answer.

IMPORTANT RULES:
1. The answer must be fully supported by the supplied chunks.
2. Do not use external knowledge.
3. Prefer questions that require combining information
   from multiple chunks when possible.
4. If the question needs only one chunk, use only that chunk ID.
5. Select supporting_chunk_ids ONLY from the supplied IDs.
6. Do not invent or modify chunk IDs.
7. Avoid vague questions.
8. The expected answer should be informative but concise.
9. If the chunks do not contain enough useful information
   to create a reliable question, return null.

Available chunk IDs:
{json.dumps(allowed_ids, ensure_ascii=False)}

Chunks:
{chunks_text}

Return ONLY valid JSON in this format:

{{
  "question": "A natural question",
  "expected_answer": "A correct answer based only on the chunks",
  "supporting_chunk_ids": ["exact_chunk_id_1", "exact_chunk_id_2"]
}}

If no reliable question can be generated, return:

{{
  "question": null,
  "expected_answer": null,
  "supporting_chunk_ids": []
}}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You generate accurate RAG evaluation data. Return valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=1000,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content

    return json.loads(content)


# =========================
# MAIN
# =========================

def generate_dataset():

    print("Loading chunks from ChromaDB...")

    chunks = get_all_chunks(source="Building the Future Egypt’s Political and Social Achievements in 2024 and the Road to 2025.pdf")

    if not chunks:
        print("No chunks found in ChromaDB.")
        return

    print(f"Total chunks: {len(chunks)}")

    groups = group_chunks(chunks)

    print(f"Total groups: {len(groups)}")

    dataset = []
    seen_questions = set()

    for index, group in enumerate(groups, start=1):

        try:
            result = generate_question(group)

            question = result.get("question")
            answer = result.get("expected_answer")
            relevant_ids = result.get("supporting_chunk_ids", [])

            # Validate generated output
            allowed_ids = {chunk["id"] for chunk in group}

            relevant_ids = list(dict.fromkeys(
                chunk_id
                for chunk_id in relevant_ids
                if chunk_id in allowed_ids
            ))

            if not question or not answer or not relevant_ids:
                print(f"Group {index}: Skipped")
                continue

            question_key = question.strip().lower()

            if question_key in seen_questions:
                print(f"Group {index}: Duplicate question")
                continue

            seen_questions.add(question_key)

            dataset.append({
                "question": question,
                "expected_answer": answer,
                "relevant_chunk_ids": relevant_ids
            })

            # Save progress after every successful question
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    dataset,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

            print(
                f"Question {len(dataset)} generated "
                f"(Group {index}/{len(groups)})"
            )

        except Exception as e:
            print(f"Group {index}: Error - {e}")

    print("\n========== GENERATION COMPLETE ==========")
    print(f"Total generated questions: {len(dataset)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_dataset()
