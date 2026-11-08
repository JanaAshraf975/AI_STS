
import json
import time

from vectoredb import search
from llm import ask_llm, GENERATOR_MODEL, ROUTER_MODEL, JUDGE_MODEL
from web_search import web_search
from logger_config import setup_logger

logger = setup_logger()


SYSTEM = (
    "Answer using ONLY the provided document context. "
    "Cite sources like [1], [2] after the claims they support. "
    "If the context does not contain the answer, reply exactly: "
    "\"I could not find this in the documents.\""
)


def parse_json(text):
    """Parse a JSON response returned by the LLM."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("LLM returned invalid JSON")
        raise ValueError(f"Invalid JSON returned by LLM: {text}") from e


def route_question(question):
    prompt = f"""
You are a router for a PDF question-answering system.

Decide whether the question should be answered using the user's PDF documents.

Rules:
1. Choose "search" if the question asks about a topic, concept,
   definition, explanation, example, or fact that might be found in the PDFs.
2. Choose "search" if you are unsure whether the answer exists in the PDFs.
3. Choose "direct" ONLY for simple general conversation,
   such as greetings or thanks.
4. Do not answer the question. Only classify it.

Return JSON only:
{{"route": "search"}} or {{"route": "direct"}}

Question: {question}
"""

    result = ask_llm(
        prompt,
        model=ROUTER_MODEL,
        temperature=0,
        json_mode=True,
    )

    decision = parse_json(result)
    route = decision.get("route", "search")

    if route not in ("search", "direct"):
        logger.warning("Invalid route %r; defaulting to search", route)
        route = "search"

    logger.info("Question routed to: %s", route)
    return route


def classify_retrieval(question):
    prompt = f"""
Decide how many retrieval steps are needed to answer
the question using PDF documents.

Rules:
- "single-hop": one retrieval step should be enough.
- "multi-hop": answering requires information from
  multiple aspects that need separate retrieval steps.

Return JSON only:
{{"hop_type": "single-hop"}}
or
{{"hop_type": "multi-hop"}}

Question: {question}
"""

    result = ask_llm(
        prompt,
        model=ROUTER_MODEL,
        temperature=0,
        json_mode=True,
    )

    decision = parse_json(result)
    hop_type = decision.get("hop_type", "single-hop")

    if hop_type not in ("single-hop", "multi-hop"):
        logger.warning(
            "Invalid hop type %r; defaulting to single-hop",
            hop_type
        )
        return "single-hop"

    logger.info("Retrieval classified as: %s", hop_type)
    return hop_type


def decompose_question(question):
    prompt = f"""
Break the question into 2 or 3 independent sub-questions
that together help answer the original question using PDF documents.

Return JSON only:
{{"sub_questions": ["question 1", "question 2"]}}

Question:
{question}
"""

    result = ask_llm(
        prompt,
        model=ROUTER_MODEL,
        temperature=0,
        json_mode=True,
    )

    data = parse_json(result)
    questions = data.get("sub_questions", [])

    if not isinstance(questions, list):
        logger.warning(
            "Sub-questions were not a list; using original question"
        )
        return [question]

    questions = [
        q.strip()
        for q in questions
        if isinstance(q, str) and q.strip()
    ]

    selected = questions[:3] if questions else [question]

    logger.info(
        "Question decomposed into %d sub-questions",
        len(selected)
    )

    return selected


def retrieve_chunks(question, k=5):
    hop_type = classify_retrieval(question)

    if hop_type == "single-hop":
        logger.debug("Starting single-hop retrieval (k=%d)", k)

        chunks = search(question, k=k)

        logger.info(
            "Single-hop retrieved %d chunks",
            len(chunks)
        )

        return chunks, hop_type

    sub_questions = decompose_question(question)

    logger.debug(
        "Starting multi-hop retrieval with %d searches",
        len(sub_questions)
    )

    all_chunks = []
    seen = set()

    for sub_question in sub_questions:
        chunks = search(sub_question, k=k)

        logger.debug(
            "Sub-question retrieval returned %d chunks",
            len(chunks)
        )

        for chunk in chunks:
            key = chunk.get("id") or (
                chunk.get("source"),
                chunk.get("page"),
                chunk.get("text"),
            )

            if key not in seen:
                seen.add(key)
                all_chunks.append(chunk)

    logger.info(
        "Multi-hop retrieved %d unique chunks",
        len(all_chunks)
    )

    return all_chunks, hop_type


def evaluate_retrieval(question, chunks):
    """Judge whether the retrieved chunks are relevant to the question."""

    if not chunks:
        logger.warning("Retrieval evaluation received no chunks")

        return {
            "relevant": False,
            "rewritten_query": question,
        }

    context = "\n\n".join(c["text"] for c in chunks)

    prompt = f"""
Evaluate whether the retrieved passages contain useful information
for answering the question.

Return JSON only, with these keys:
- "relevant": true or false
- "rewritten_query": a better search query if the passages are
  irrelevant or insufficient; otherwise use the original question.

Do not judge relevance only by exact keyword matches.
Focus on meaning and whether the passages help answer the question.

Question:
{question}

Retrieved passages:
{context}
"""

    result = ask_llm(
        prompt,
        model=JUDGE_MODEL,
        temperature=0,
        json_mode=True,
    )

    decision = parse_json(result)

    relevant = decision.get("relevant") is True

    logger.info("Retrieval relevance: %s", relevant)

    return {
        "relevant": relevant,
        "rewritten_query": decision.get("rewritten_query") or question,
    }


def build_prompt(question, chunks):
    context = "\n\n".join(
        f"[{i}] ({c['source']} p.{c['page']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )

    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def build_web_context(results):
    return "\n\n".join(
        f"[{i}] {result['title']}\n"
        f"URL: {result['url']}\n"
        f"{result['text']}"
        for i, result in enumerate(results, start=1)
    )


def answer_from_web(question):
    logger.warning("Falling back to web search")

    results = web_search(question, max_results=5)

    logger.info(
        "Web search returned %d results",
        len(results)
    )

    if not results:
        logger.warning("No web search results available")

        return {
            "answer": "I could not find a useful answer in the documents or on the web.",
            "sources": [],
            "meta": None,
        }

    context = build_web_context(results)

    prompt = f"""
Answer the question using the web search results below.

Question:
{question}

Web search results:
{context}

Rules:
- Use only information supported by the search results.
- Cite sources using [1], [2], etc.
- If the results do not provide enough information,
  say that you could not find a reliable answer.
"""

    text, meta = ask_llm(
        prompt,
        model=GENERATOR_MODEL,
        temperature=0,
        return_meta=True,
    )

    logger.info("Web-based answer generated")

    return {
        "answer": text,
        "sources": [result["url"] for result in results],
        "meta": meta,
    }


def format_history(history):
    if not history:
        return ""

    return "\n".join(
        f"{m['role']}: {m['content']}"
        for m in history[-6:]
    )


def rewrite_query(question, history=None):
    if not history:
        logger.debug(
            "No conversation history; query rewrite skipped"
        )
        return question

    logger.debug(
        "Rewriting query using %d history messages",
        min(len(history), 6)
    )

    conversation = format_history(history)

    prompt = f"""
Given the conversation history and the latest user question,
rewrite the latest question as a standalone question for document search.

Rules:
- Keep the same language.
- Use previous messages to understand unclear references.
- Do not answer the question.
- Return only the rewritten question.

Conversation:
{conversation}

Latest question: {question}
"""

    rewritten = ask_llm(
        prompt,
        model=ROUTER_MODEL,
        temperature=0,
    )

    logger.info("Standalone question rewrite completed")

    return rewritten.strip() or question


def answer(question, k=5, history=None):
    start_time = time.perf_counter()

    logger.info("RAG request started (k=%d)", k)

    history = history or []

    standalone_question = rewrite_query(question, history)

    route = route_question(standalone_question)

    # 1. Direct answer
    if route == "direct":
        logger.debug("Generating direct answer")

        text, meta = ask_llm(
            standalone_question,
            model=GENERATOR_MODEL,
            return_meta=True,
        )

        logger.info(
            "RAG completed | route=direct | duration=%.2fs",
            time.perf_counter() - start_time
        )

        return {
            "question": question,
            "answer": text,
            "contexts": [],
            "chunk_ids": [],
            "sources": [],
            "meta": meta,
            "route": "direct",
            "hop_type": None,
            "retried": False,
        }

    # 2. Single-Hop or Multi-Hop retrieval
    chunks, hop_type = retrieve_chunks(
        standalone_question,
        k=k
    )

    evaluation = evaluate_retrieval(
        standalone_question,
        chunks
    )

    retried = False

    # 3. Retry if retrieval is weak
    if not evaluation["relevant"]:
        logger.warning(
            "Initial retrieval judged insufficient; retrying"
        )

        new_query = evaluation["rewritten_query"]

        retry_chunks = search(new_query, k=k)

        logger.info(
            "Retry retrieved %d chunks",
            len(retry_chunks)
        )

        retry_evaluation = evaluate_retrieval(
            standalone_question,
            retry_chunks,
        )

        retried = True

        logger.info(
            "Retry relevance: %s",
            retry_evaluation["relevant"]
        )

        if retry_evaluation["relevant"]:
            chunks = retry_chunks

        else:
            # 4. Web Search Fallback
            logger.warning(
                "Retry insufficient; switching to web fallback"
            )

            web_result = answer_from_web(
                standalone_question
            )

            logger.info(
                "RAG completed | route=web_search | "
                "hop=%s | retried=True | duration=%.2fs",
                hop_type,
                time.perf_counter() - start_time
            )

            return {
                "question": question,
                "answer": web_result["answer"],
                "contexts": [],
                "chunk_ids": [],
                "sources": web_result["sources"],
                "meta": web_result["meta"],
                "route": "web_search",
                "hop_type": hop_type,
                "retried": True,
            }

    # 5. Generate answer from retrieved PDF chunks
    logger.debug(
        "Generating PDF-based answer from %d chunks",
        len(chunks)
    )

    text, meta = ask_llm(
        build_prompt(standalone_question, chunks),
        model=GENERATOR_MODEL,
        system=SYSTEM,
        return_meta=True,
    )

    logger.info(
        "RAG completed | route=search | hop=%s | "
        "retried=%s | chunks=%d | duration=%.2fs",
        hop_type,
        retried,
        len(chunks),
        time.perf_counter() - start_time
    )

    return {
        "question": question,
        "answer": text,
        "contexts": [c["text"] for c in chunks],
        "chunk_ids": [c["id"] for c in chunks],
        "sources": [
            (c["source"], c["page"])
            for c in chunks
        ],
        "meta": meta,
        "route": "search",
        "hop_type": hop_type,
        "retried": retried,
    }


if __name__ == "__main__":
    history = []

    logger.info("Adaptive RAG application started")

    while True:
        q = input("\nQuestion (or q to quit): ").strip()

        if q.lower() == "q":
            logger.info("Adaptive RAG application stopped")
            break

        if not q:
            continue

        try:
            result = answer(q, history=history)

            print("\nRoute:", result["route"])
            print("Hop type:", result.get("hop_type"))
            print("Retried:", result["retried"])
            print("\nAnswer:")
            print(result["answer"])
            print("\nSources:", result["sources"])
            print("Meta:", result["meta"])

            # Save conversation in history
            history.append({
                "role": "user",
                "content": q
            })

            history.append({
                "role": "assistant",
                "content": result["answer"]
            })

        except Exception as e:
            logger.exception(
                "Error while processing RAG question"
            )
            print(f"Error: {e}")
