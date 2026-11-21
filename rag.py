
import json
import logging
import time
from functools import wraps

from vectoredb import search, get_all_chunks , save_summary , get_summary
from llm import (
    ask_llm,
    GENERATOR_MODEL,
    ROUTER_MODEL,
    JUDGE_MODEL,
    reset_usage_totals,
    get_usage_totals,
)
from web_search import web_search
from logger_config import logger, log_event, start_request, end_request


# ==============================
# Logging Helpers
# ==============================

def elapsed_ms(start):
    return round((time.perf_counter() - start) * 1000, 2)


def chunk_details(chunks):
    return [
        {
            "id": c.get("id"),
            "source": c.get("source"),
            "page": c.get("page"),
            "similarity": round(c["score"], 4)
            if isinstance(c.get("score"), (int, float)) else None
        }
        for c in chunks
    ]


def log_chunks(event, chunks, start, **details):
    log_event(
        logging.INFO,
        "retrieval",
        event,
        "Vector retrieval completed",
        chunks_retrieved=len(chunks),
        chunks=chunk_details(chunks),
        latency_ms=elapsed_ms(start),
        **details
    )


# ==============================
# Request Tracking
# ==============================

def traced_request(func):
    """Track requests, including calls from evaluation scripts."""

    @wraps(func)
    def wrapper(question, *args, **kwargs):
        token = start_request()
        start = time.perf_counter()

        # Reset usage before starting a new RAG request
        reset_usage_totals()

        try:
            log_event(
                logging.INFO,
                "pipeline",
                "request_started",
                "New RAG request received",
                question_length=len(question)
            )

            result = func(question, *args, **kwargs)

            # Collect usage from all LLM calls in this request
            usage = get_usage_totals()

            # Attach usage information to the result
            result["usage"] = usage

            log_event(
                logging.INFO,
                "pipeline",
                "request_completed",
                "RAG request completed successfully",
                route=result.get("route"),
                hop_type=result.get("hop_type"),
                retried=result.get("retried"),
                sources_count=len(result.get("sources") or []),
                chunks_count=len(result.get("chunk_ids") or []),
                model_metadata=result.get("meta"),
                total_llm_usage=usage,
                latency_ms=elapsed_ms(start)
            )

            return result

        except Exception:
            logger.exception(
                "RAG request failed",
                extra={
                    "component": "pipeline",
                    "event": "request_failed",
                    "details": {
                        "latency_ms": elapsed_ms(start)
                    }
                }
            )
            raise

        finally:
            end_request(token)

    return wrapper


SYSTEM = (
    "Answer using ONLY the provided document context. "
    "Cite sources like [1], [2] after the claims they support. "
    "If the context does not contain the answer, reply exactly: "
    "\"I could not find this in the documents.\""
)


# ==============================
# JSON Parser
# ==============================

def parse_json(text):
    """Parse a JSON response returned by the LLM."""
    try:
        return json.loads(text)

    except json.JSONDecodeError:
        logger.exception(
            "Failed to parse JSON response from LLM",
            extra={
                "component": "llm",
                "event": "json_parse_failed",
                "details": {"response_length": len(text)}
            }
        )
        raise ValueError(f"Invalid JSON returned by LLM: {text}")


# ==============================
# Question Router
# ==============================

def route_question(question, available_sources):
    start = time.perf_counter()

    prompt = f"""
You are an intelligent router for a PDF question-answering system.

Available PDF documents:
{json.dumps(available_sources, ensure_ascii=False)}

Understand the user's intention, even if they use different
wording to request a summary.

Choose one route:

1. "summarize_all":
   When the user wants a summary of ALL documents,
   every PDF, or the entire document collection.

2. "summarize_one":
   When the user wants to summarize ONE specific PDF,
   or asks for an overview of a particular document.

   Identify which PDF they mean from the available documents.
   Return its EXACT filename in "source".

3. "search":
   When the user asks a question about information,
   facts, concepts, definitions, or explanations in the PDFs.

4. "direct":
   ONLY for simple conversation such as greetings or thanks.

Important rules:
- Understand natural language and different ways of expressing
  the same intention.
- Do not confuse asking about information in a PDF with
  requesting a summary of that PDF.
- If the user explicitly asks for all documents, choose
  "summarize_all".
- If the user refers to a specific document, choose
  "summarize_one".
- Never invent a filename.
- If the requested document cannot be identified, use null
  for source.
- Do not answer the question.

Return JSON only:

{{"route": "summarize_all", "source": null}}

OR

{{"route": "summarize_one", "source": "exact_filename.pdf"}}

OR

{{"route": "search", "source": null}}

OR

{{"route": "direct", "source": null}}

Question:
{question}
"""

    result = ask_llm(
        prompt,
        model=ROUTER_MODEL,
        temperature=0,
        json_mode=True,
    )

    decision = parse_json(result)

    route = decision.get("route", "search")
    source = decision.get("source")

    if route not in (
        "search",
        "direct",
        "summarize_all",
        "summarize_one"
    ):
        log_event(
            logging.WARNING,
            "router",
            "invalid_route",
            "Invalid route returned by LLM",
            returned_route=route,
            fallback_route="search"
        )
        route = "search"

    if route == "summarize_one" and source:
        source = source.strip()
    else:
        source = None

    log_event(
        logging.INFO,
        "router",
        "route_selected",
        "Question routing completed",
        route=route,
        selected_source=source,
        available_sources=len(available_sources),
        model=ROUTER_MODEL,
        latency_ms=elapsed_ms(start)
    )

    return {
        "route": route,
        "source": source,
    }


# ==============================
# Retrieval Classification
# ==============================

def classify_retrieval(question):
    start = time.perf_counter()

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
        log_event(
            logging.WARNING,
            "router",
            "invalid_hop_type",
            "Invalid retrieval classification",
            returned_hop_type=hop_type
        )
        hop_type = "single-hop"

    log_event(
        logging.INFO,
        "router",
        "retrieval_classified",
        "Retrieval strategy selected",
        hop_type=hop_type,
        model=ROUTER_MODEL,
        latency_ms=elapsed_ms(start)
    )

    return hop_type


# ==============================
# Question Decomposition
# ==============================

def decompose_question(question):
    start = time.perf_counter()

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
        questions = []

    questions = [
        q.strip()
        for q in questions
        if isinstance(q, str) and q.strip()
    ]

    questions = questions[:3] if questions else [question]

    log_event(
        logging.INFO,
        "retrieval",
        "question_decomposed",
        "Multi-hop question decomposed",
        sub_questions_count=len(questions),
        sub_question_lengths=[len(q) for q in questions],
        latency_ms=elapsed_ms(start)
    )

    return questions


# ==============================
# Retrieval
# ==============================

def retrieve_chunks(question, k=5, collection_name="docs"):
    start = time.perf_counter()

    hop_type = classify_retrieval(question)

    if hop_type == "single-hop":
        search_start = time.perf_counter()

        chunks = search(question, k=k)

        log_chunks(
            "single_hop_completed",
            chunks,
            search_start,
            top_k=k,
            hop_type=hop_type
        )

        return chunks, hop_type

    sub_questions = decompose_question(question)

    all_chunks = []
    seen = set()
    total_hits = 0

    for index, sub_question in enumerate(sub_questions, start=1):
        sub_start = time.perf_counter()

        chunks = search(sub_question, k=k)
        total_hits += len(chunks)

        log_chunks(
            "sub_question_retrieved",
            chunks,
            sub_start,
            sub_question_index=index,
            top_k=k
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

    log_event(
        logging.INFO,
        "retrieval",
        "multi_hop_completed",
        "Multi-hop retrieval completed",
        sub_questions_count=len(sub_questions),
        total_hits=total_hits,
        unique_chunks=len(all_chunks),
        duplicates_removed=total_hits - len(all_chunks),
        chunks=chunk_details(all_chunks),
        latency_ms=elapsed_ms(start)
    )

    return all_chunks, hop_type


# ==============================
# Retrieval Evaluation
# ==============================

def evaluate_retrieval(question, chunks):
    """Judge whether the retrieved chunks are relevant to the question."""

    start = time.perf_counter()

    if not chunks:
        log_event(
            logging.WARNING,
            "evaluation",
            "retrieval_empty",
            "No retrieved chunks available",
            relevant=False,
            chunks_count=0
        )

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
    rewritten_query = decision.get("rewritten_query") or question

    log_event(
        logging.INFO,
        "evaluation",
        "retrieval_evaluated",
        "Retrieved context evaluated",
        relevant=relevant,
        chunks_evaluated=len(chunks),
        query_rewritten=rewritten_query != question,
        rewritten_query_length=len(rewritten_query),
        judge_model=JUDGE_MODEL,
        latency_ms=elapsed_ms(start)
    )

    return {
        "relevant": relevant,
        "rewritten_query": rewritten_query,
    }

def retrieve_summaries(chunks):
    summaries = []
    seen_sources = set()

    for chunk in chunks:
        source = chunk.get("source")

        if source and source not in seen_sources:
            seen_sources.add(source)

            summary = get_summary(source)

            if summary:
                summaries.append(summary)

    return summaries


# ==============================
# Prompt Builder
# ==============================

def build_prompt(question, chunks, summaries=None):
    chunk_context = "\n\n".join(
        f"[{i}]\n{c['text'] if isinstance(c, dict) else c}"
        for i, c in enumerate(chunks, start=1)
    )

    summary_context = "\n\n".join(
        f"[S{i}] ({s['source']} - PDF Summary)\n{s['text']}"
        for i, s in enumerate(summaries or [], start=1)
    )

    prompt = (
        f"PDF Summaries:\n{summary_context}\n\n"
        f"Original Retrieved Chunks:\n{chunk_context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    return prompt
# Web Search
# ==============================

def build_web_context(results):
    return "\n\n".join(
        f"[{i}] {result['title']}\n"
        f"URL: {result['url']}\n"
        f"{result['text']}"
        for i, result in enumerate(results, start=1)
    )


def answer_from_web(question):
    start = time.perf_counter()

    log_event(
        logging.WARNING,
        "web_search",
        "fallback_started",
        "Starting web search fallback",
        max_results=5
    )

    search_start = time.perf_counter()

    # Original behavior: web_search exceptions are not swallowed
    results = web_search(question, max_results=5)

    log_event(
        logging.INFO,
        "web_search",
        "search_completed",
        "Web search completed",
        results_count=len(results),
        search_latency_ms=elapsed_ms(search_start)
    )

    if not results:
        log_event(
            logging.WARNING,
            "web_search",
            "no_results",
            "Web search returned no results",
            latency_ms=elapsed_ms(start)
        )

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

    log_event(
        logging.INFO,
        "web_search",
        "web_answer_generated",
        "Web fallback answer generated",
        results_count=len(results),
        answer_chars=len(text),
        model_metadata=meta,
        latency_ms=elapsed_ms(start)
    )

    return {
        "answer": text,
        "sources": [result["url"] for result in results],
        "meta": meta,
    }


# ==============================
# PDF Summarization
# ==============================


def summarize_all_pdfs(selected_sources=None, max_chars=12000):
    start = time.perf_counter()

    log_event(
        logging.INFO,
        "summarization",
        "summarization_started",
        "Starting PDF summarization",
        selected_sources=selected_sources,
        max_chars=max_chars
    )

    chunks = get_all_chunks()

    if not chunks:
        log_event(
            logging.WARNING,
            "summarization",
            "no_documents",
            "No PDF documents found in database"
        )

        return "No PDF documents found in the database.", []

    pdfs = {}

    for chunk in chunks:
        source = chunk["source"]

        if source not in pdfs:
            pdfs[source] = []

        pdfs[source].append(chunk)

    if selected_sources:
        pdfs = {
            source: pdf_chunks
            for source, pdf_chunks in pdfs.items()
            if source in selected_sources
        }

    if not pdfs:
        log_event(
            logging.WARNING,
            "summarization",
            "source_not_found",
            "Requested PDF was not found",
            selected_sources=selected_sources
        )

        return "The requested PDF was not found.", []

    all_pdf_summaries = []

    for source, pdf_chunks in pdfs.items():
        pdf_start = time.perf_counter()

        log_event(
            logging.INFO,
            "summarization",
            "pdf_processing_started",
            "PDF summarization started",
            source=source,
            chunks_count=len(pdf_chunks)
        )

        pdf_chunks.sort(key=lambda c: c["page"])

        batches = []
        current_batch = ""

        for chunk in pdf_chunks:
            text = (
                f"Page {chunk['page']}:\n"
                f"{chunk['text']}\n\n"
            )

            if (
                current_batch
                and len(current_batch) + len(text) > max_chars
            ):
                batches.append(current_batch)
                current_batch = ""

            current_batch += text

        if current_batch:
            batches.append(current_batch)

        log_event(
            logging.INFO,
            "summarization",
            "pdf_batches_created",
            "PDF divided into batches",
            source=source,
            batches_count=len(batches),
            max_chars=max_chars
        )

        partial_summaries = []

        for i, batch in enumerate(batches, 1):
            batch_start = time.perf_counter()

            prompt = f"""
Summarize the following content from the PDF: {source}

This is part {i} of the document.

Rules:
- Include the main ideas and important details.
- Preserve important names, numbers, facts, and specifications.
- Do not add information that is not in the content.
- Avoid unnecessary repetition.

Content:
{batch}

Summary:
"""

            partial_summary = ask_llm(
                prompt,
                model=GENERATOR_MODEL,
                temperature=0,
            )

            partial_summaries.append(partial_summary.strip())

            log_event(
                logging.DEBUG,
                "summarization",
                "batch_completed",
                "PDF summary batch completed",
                source=source,
                batch_number=i,
                total_batches=len(batches),
                batch_chars=len(batch),
                summary_chars=len(partial_summary),
                latency_ms=elapsed_ms(batch_start)
            )

        final_prompt = f"""
Create one clear and organized summary of this PDF.

PDF: {source}

Partial summaries:
{chr(10).join(partial_summaries)}

Rules:
- Combine related ideas.
- Remove repeated information.
- Preserve important facts, names, numbers, and specifications.
- Do not invent information.

Final Summary:
"""

        final_start = time.perf_counter()

        final_summary = ask_llm(
            final_prompt,
            model=GENERATOR_MODEL,
            temperature=0,
        )

        final_summary = final_summary.strip()

        # Save the final summary in ChromaDB
        save_summary(source, final_summary)

        all_pdf_summaries.append(
            f"## {source}\n{final_summary}"
        )

        log_event(
            logging.INFO,
            "summarization",
            "pdf_completed",
            "PDF summarization completed",
            source=source,
            batches_count=len(batches),
            summary_chars=len(final_summary),
            summary_saved=True,
            final_summary_latency_ms=elapsed_ms(final_start),
            latency_ms=elapsed_ms(pdf_start)
        )

    log_event(
        logging.INFO,
        "summarization",
        "summarization_completed",
        "PDF summarization completed",
        pdf_count=len(pdfs),
        latency_ms=elapsed_ms(start)
    )

    return "\n\n".join(all_pdf_summaries), list(pdfs.keys())

# ==============================
# Conversation Memory
# ==============================

def rewrite_query(question, history, conversation_summary):
    start = time.perf_counter()

    if not history and not conversation_summary:
        log_event(
            logging.DEBUG,
            "memory",
            "rewrite_skipped",
            "No conversation history to rewrite query"
        )
        return question

    recent_conversation = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in history
    )

    prompt = f"""
Rewrite the latest user question as a standalone question
for document search.

Older Conversation Summary:
{conversation_summary if conversation_summary else "No summary."}

Last 3 Questions and Answers:
{recent_conversation if recent_conversation else "No history."}

Latest Question:
{question}

Rules:
- Keep the same language.
- Use the summary and recent conversation to understand references.
- Resolve unclear references such as it, its, this, and that.
- Do not answer the question.
- Return only the rewritten question.
"""

    rewritten = ask_llm(
        prompt,
        model=ROUTER_MODEL,
        temperature=0,
    )

    rewritten = rewritten.strip() or question

    log_event(
        logging.INFO,
        "memory",
        "query_rewritten",
        "Conversation-aware query rewrite completed",
        changed=rewritten != question,
        history_messages=len(history),
        summary_available=bool(conversation_summary),
        latency_ms=elapsed_ms(start)
    )

    return rewritten


def update_conversation_summary(summary, old_messages):
    start = time.perf_counter()

    old_conversation = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in old_messages
    )

    prompt = f"""
Update the conversation summary using the older messages.

Previous Summary:
{summary if summary else "No previous summary."}

New Older Messages:
{old_conversation}

Rules:
- Preserve important information from previous discussions.
- Keep important names, product names, prices, and specifications.
- Preserve important user preferences and decisions.
- Keep the information in chronological order.
- Do not invent information.
- Keep the summary concise.

Return only the updated summary.
"""

    updated_summary = ask_llm(
        prompt,
        model=GENERATOR_MODEL,
        temperature=0,
    )

    log_event(
        logging.INFO,
        "memory",
        "summary_updated",
        "Conversation summary updated",
        old_messages_count=len(old_messages),
        previous_summary_chars=len(summary),
        updated_summary_chars=len(updated_summary),
        latency_ms=elapsed_ms(start)
    )

    return updated_summary.strip() or summary


# ==============================
# Main Answer Function
# ==============================

@traced_request
def answer(question, k=5, history=None, conversation_summary="", collection_name="docs"):

    history = history or []

    log_event(
        logging.DEBUG,
        "pipeline",
        "processing_started",
        "Processing question",
        history_messages=len(history),
        top_k=k,
        has_summary=bool(conversation_summary)
    )

    # 1. Rewrite question using conversation history
    standalone_question = rewrite_query(
        question,
        history,
        conversation_summary
    )

    # 2. Get available PDF documents
    source_start = time.perf_counter()

    available_sources = list({
        chunk["source"]
        for chunk in get_all_chunks()
    })

    log_event(
        logging.INFO,
        "pipeline",
        "sources_loaded",
        "Available PDF sources loaded",
        sources_count=len(available_sources),
        latency_ms=elapsed_ms(source_start)
    )

    # 3. Route the question
    decision = route_question(
        standalone_question,
        available_sources
    )

    route = decision["route"]

    log_event(
        logging.INFO,
        "pipeline",
        "route_selected",
        "Question route selected",
        route=route,
        selected_source=decision["source"]
    )

    # 4. Summarize ALL PDFs
    if route == "summarize_all":

        summary, sources = summarize_all_pdfs()

        return {
            "question": question,
            "answer": summary,
            "contexts": [],
            "summaries": [],
            "chunk_ids": [],
            "sources": sources,
            "summary_sources": [],
            "meta": None,
            "route": "summarize_all",
            "hop_type": None,
            "retried": False,
        }

    # 5. Summarize ONE specific PDF
    if route == "summarize_one":

        selected_source = decision["source"]

        matched_source = next(
            (
                source for source in available_sources
                if selected_source
                and source.lower() == selected_source.lower()
            ),
            None
        )

        if not matched_source:

            log_event(
                logging.WARNING,
                "router",
                "source_not_identified",
                "Could not identify requested PDF",
                selected_source=selected_source,
                available_sources=available_sources
            )

            return {
                "question": question,
                "answer": (
                    "Which PDF would you like me to summarize?\n\n"
                    "Available PDFs:\n"
                    + "\n".join(
                        f"- {source}"
                        for source in available_sources
                    )
                ),
                "contexts": [],
                "summaries": [],
                "chunk_ids": [],
                "sources": available_sources,
                "summary_sources": [],
                "meta": None,
                "route": "summarize_one",
                "hop_type": None,
                "retried": False,
            }

        summary, sources = summarize_all_pdfs(
            selected_sources=[matched_source]
        )

        return {
            "question": question,
            "answer": summary,
            "contexts": [],
            "summaries": [],
            "chunk_ids": [],
            "sources": sources,
            "summary_sources": [],
            "meta": None,
            "route": "summarize_one",
            "hop_type": None,
            "retried": False,
        }

    # 6. Direct answer
    if route == "direct":

        generation_start = time.perf_counter()

        text, meta = ask_llm(
            standalone_question,
            model=GENERATOR_MODEL,
            return_meta=True,
        )

        log_event(
            logging.INFO,
            "generation",
            "direct_answer_generated",
            "Direct answer generated",
            answer_chars=len(text),
            model_metadata=meta,
            latency_ms=elapsed_ms(generation_start)
        )

        return {
            "question": question,
            "answer": text,
            "contexts": [],
            "summaries": [],
            "chunk_ids": [],
            "sources": [],
            "summary_sources": [],
            "meta": meta,
            "route": "direct",
            "hop_type": None,
            "retried": False,
        }

    # 7. Single-Hop or Multi-Hop retrieval
    chunks, hop_type = retrieve_chunks(
        standalone_question,
        k=k, collection_name=collection_name
    )

    # 8. Evaluate retrieval
    evaluation = evaluate_retrieval(
        standalone_question,
        chunks
    )

    retried = False

    # 9. Retry if retrieval is weak
    if not evaluation["relevant"]:

        log_event(
            logging.WARNING,
            "retrieval",
            "retry_started",
            "Retrieved chunks are not relevant. Retrying retrieval",
            initial_chunks_count=len(chunks)
        )

        new_query = evaluation["rewritten_query"]
        retry_start = time.perf_counter()

        retry_chunks = search(new_query, k=k)

        log_chunks(
            "retry_retrieved",
            retry_chunks,
            retry_start,
            top_k=k,
            query_changed=new_query != standalone_question
        )

        retry_evaluation = evaluate_retrieval(
            standalone_question,
            retry_chunks,
        )

        retried = True

        if retry_evaluation["relevant"]:

            chunks = retry_chunks

            log_event(
                logging.INFO,
                "retrieval",
                "retry_succeeded",
                "Retry retrieval succeeded",
                chunks_count=len(chunks)
            )

        else:

            log_event(
                logging.WARNING,
                "retrieval",
                "web_fallback_triggered",
                "Retry failed. Falling back to web search",
                retry_chunks_count=len(retry_chunks)
            )

            # 10. Web Search Fallback
            web_result = answer_from_web(standalone_question)

            return {
                "question": question,
                "answer": web_result["answer"],
                "contexts": [],
                "summaries": [],
                "chunk_ids": [],
                "sources": web_result["sources"],
                "summary_sources": [],
                "meta": web_result["meta"],
                "route": "web_search",
                "hop_type": hop_type,
                "retried": True,
            }

    # 11. Retrieve summaries for PDFs represented in final chunks
    summaries = retrieve_summaries(chunks)

    log_event(
        logging.INFO,
        "retrieval",
        "summaries_retrieved",
        "PDF summaries retrieved for generation",
        summaries_count=len(summaries),
        summary_sources=[
            s["source"] for s in summaries
        ]
    )

    # 12. Generate answer using chunks + PDF summaries
    generation_start = time.perf_counter()

    text, meta = ask_llm(
        build_prompt(
            standalone_question,
            chunks,
            summaries
        ),
        model=GENERATOR_MODEL,
        system=SYSTEM,
        return_meta=True,
    )

    log_event(
        logging.INFO,
        "generation",
        "answer_generated",
        "Answer generated successfully",
        chunks_used=len(chunks),
        summaries_used=len(summaries),
        sources=[
            {"source": c["source"], "page": c["page"]}
            for c in chunks
        ],
        summary_sources=[
            s["source"] for s in summaries
        ],
        answer_chars=len(text),
        model_metadata=meta,
        latency_ms=elapsed_ms(generation_start)
    )

    return {
        "question": question,
        "answer": text,
        "contexts": [c["text"] for c in chunks],
        "summaries": [
            s["text"] for s in summaries
        ],
        "chunk_ids": [c["id"] for c in chunks],
        "sources": [
            (c["source"], c["page"])
            for c in chunks
        ],
        "summary_sources": [
            s["source"] for s in summaries
        ],
        "meta": meta,
        "route": "search",
        "hop_type": hop_type,
        "retried": retried,
    }

# ==============================
# Main Application
# ==============================

if __name__ == "__main__":

    history = []
    conversation_summary = ""

    log_event(
        logging.INFO,
        "application",
        "application_started",
        "Adaptive RAG application started"
    )

    while True:

        q = input("\nQuestion (or q to quit): ").strip()

        if q.lower() == "q":
            break

        if not q:
            continue

        try:

            result = answer(
                q,
                history=history,
                conversation_summary=conversation_summary,
            )

            print("\nAnswer:")
            print(result["answer"])

            print("\nRoute:")
            print(result["route"])

            usage = result["usage"]

            print("\n--- TOTAL LLM USAGE ---")
            print("LLM Calls:", usage["calls"])
            print("Input Tokens:", usage["prompt_tokens"])
            print("Output Tokens:", usage["completion_tokens"])
            print("Total Tokens:", usage["total_tokens"])

            print("\n--- TOTAL ESTIMATED COST ---")

            for label, key in [
                ("Input Cost", "input_cost_usd"),
                ("Output Cost", "output_cost_usd"),
                ("Total Cost", "total_cost_usd"),
            ]:

                value = usage[key]

                formatted = (
                    f"${value:.10f}"
                    if value is not None
                    else "Unavailable"
                )

                print(f"{label}: {formatted}")

            history.append({
                "role": "user",
                "content": q,
            })

            history.append({
                "role": "assistant",
                "content": result["answer"],
            })

            if len(history) > 6:

                old_messages = history[:-6]

                conversation_summary = update_conversation_summary(
                    conversation_summary,
                    old_messages,
                )

                history = history[-6:]

            print("\nHop Type:")
            print(result["hop_type"])

            print("\nCurrent Summary:")
            print(conversation_summary)

            print("\nLast 3 Questions and Answers:")
            print(history)

        except Exception:

            logger.exception("Error while processing question")
            print("An error occurred. Check the log file.")

    log_event(
        logging.INFO,
        "application",
        "application_stopped",
        "Adaptive RAG application stopped"
    )
