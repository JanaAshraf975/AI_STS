
import json

from vectoredb import search, get_all_chunks
from llm import ask_llm, GENERATOR_MODEL, ROUTER_MODEL, JUDGE_MODEL
from web_search import web_search
from logger_config import logger


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
        logger.exception("Failed to parse JSON response from LLM")
        raise ValueError(f"Invalid JSON returned by LLM: {text}")


# ==============================
# Question Router
# ==============================

def route_question(question, available_sources):

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
        route = "search"

    if route == "summarize_one" and source:
        source = source.strip()
    else:
        source = None

    logger.info(f"Question routed to: {route}")

    if source:
        logger.info(f"Selected PDF source: {source}")

    return {
        "route": route,
        "source": source,
    }


# ==============================
# Retrieval Classification
# ==============================

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
        hop_type = "single-hop"

    logger.info(f"Retrieval type: {hop_type}")

    return hop_type


# ==============================
# Question Decomposition
# ==============================

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
        return [question]

    questions = [
        q.strip()
        for q in questions
        if isinstance(q, str) and q.strip()
    ]

    questions = questions[:3] if questions else [question]

    logger.info(f"Question decomposed into {len(questions)} sub-questions")

    return questions


# ==============================
# Retrieval
# ==============================

def retrieve_chunks(question, k=5):

    hop_type = classify_retrieval(question)

    if hop_type == "single-hop":

        chunks = search(question, k=k)

        logger.info(
            f"Single-hop retrieval: {len(chunks)} chunks retrieved"
        )

        return chunks, hop_type

    sub_questions = decompose_question(question)

    logger.info(
        f"Multi-hop retrieval: {len(sub_questions)} sub-questions"
    )

    all_chunks = []
    seen = set()

    for sub_question in sub_questions:

        chunks = search(sub_question, k=k)

        logger.debug(
            f"Sub-question retrieved {len(chunks)} chunks"
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
        f"Total unique chunks retrieved: {len(all_chunks)}"
    )

    return all_chunks, hop_type


# ==============================
# Retrieval Evaluation
# ==============================

def evaluate_retrieval(question, chunks):
    """Judge whether the retrieved chunks are relevant to the question."""

    if not chunks:
        logger.warning("No chunks retrieved for evaluation")

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

    logger.info(f"Retrieval evaluation: relevant={relevant}")

    return {
        "relevant": relevant,
        "rewritten_query": decision.get("rewritten_query") or question,
    }


# ==============================
# Prompt Builder
# ==============================

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


# ==============================
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

    logger.info("Starting web search fallback")

    results = web_search(question, max_results=5)

    logger.info(f"Web search returned {len(results)} results")

    if not results:
        logger.warning("Web search returned no results")

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

    logger.info("Web answer generated successfully")

    return {
        "answer": text,
        "sources": [result["url"] for result in results],
        "meta": meta,
    }


# ==============================
# PDF Summarization
# ==============================

def summarize_all_pdfs(selected_sources=None, max_chars=12000):

    logger.info("Starting PDF summarization")

    chunks = get_all_chunks()

    if not chunks:
        logger.warning("No PDF documents found in database")

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
        logger.warning("Requested PDF was not found")

        return "The requested PDF was not found.", []

    all_pdf_summaries = []

    for source, pdf_chunks in pdfs.items():

        logger.info(f"Summarizing PDF: {source}")

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

        logger.info(
            f"PDF {source} divided into {len(batches)} batches"
        )

        partial_summaries = []

        for i, batch in enumerate(batches, 1):

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

            logger.debug(
                f"Completed summary batch {i} for {source}"
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

        final_summary = ask_llm(
            final_prompt,
            model=GENERATOR_MODEL,
            temperature=0,
        )

        all_pdf_summaries.append(
            f"## {source}\n{final_summary.strip()}"
        )

        logger.info(f"Finished summarizing PDF: {source}")

    logger.info("PDF summarization completed")

    return "\n\n".join(all_pdf_summaries), list(pdfs.keys())


# ==============================
# Conversation Memory
# ==============================

def rewrite_query(question, history, conversation_summary):

    if not history and not conversation_summary:
        logger.debug("No conversation history to rewrite query")
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

    logger.info("Question rewritten using conversation history")
    logger.debug(f"Rewritten question: {rewritten}")

    return rewritten


def update_conversation_summary(summary, old_messages):

    logger.info(
        f"Updating conversation summary using {len(old_messages)} older messages"
    )

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

    logger.info("Conversation summary updated successfully")

    return updated_summary.strip() or summary


# ==============================
# Main Answer Function
# ==============================

def answer(question, k=5, history=None, conversation_summary=""):

    history = history or []

    logger.info("New question received")
    logger.debug(f"Conversation history: {len(history)} messages")

    # 1. Rewrite question using conversation history
    standalone_question = rewrite_query(
        question,
        history,
        conversation_summary
    )

    # 2. Get available PDF documents
    available_sources = list({
        chunk["source"]
        for chunk in get_all_chunks()
    })

    logger.info(f"Available PDF documents: {len(available_sources)}")

    # 3. Route the question
    decision = route_question(
        standalone_question,
        available_sources
    )

    route = decision["route"]

    logger.info(f"Selected route: {route}")

    # 4. Summarize ALL PDFs
    if route == "summarize_all":

        summary, sources = summarize_all_pdfs()

        logger.info("All PDF summaries generated")

        return {
            "question": question,
            "answer": summary,
            "contexts": [],
            "chunk_ids": [],
            "sources": sources,
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

            logger.warning(
                f"Could not identify requested PDF: {selected_source}"
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
                "chunk_ids": [],
                "sources": available_sources,
                "meta": None,
                "route": "summarize_one",
                "hop_type": None,
                "retried": False,
            }

        summary, sources = summarize_all_pdfs(
            selected_sources=[matched_source]
        )

        logger.info(f"Summary generated for PDF: {matched_source}")

        return {
            "question": question,
            "answer": summary,
            "contexts": [],
            "chunk_ids": [],
            "sources": sources,
            "meta": None,
            "route": "summarize_one",
            "hop_type": None,
            "retried": False,
        }

    # 6. Direct answer
    if route == "direct":

        text, meta = ask_llm(
            standalone_question,
            model=GENERATOR_MODEL,
            return_meta=True,
        )

        logger.info("Direct answer generated successfully")

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

    # 7. Single-Hop or Multi-Hop retrieval
    chunks, hop_type = retrieve_chunks(standalone_question, k=k)

    # 8. Evaluate retrieval
    evaluation = evaluate_retrieval(standalone_question, chunks)
    retried = False

    # 9. Retry if retrieval is weak
    if not evaluation["relevant"]:

        logger.warning("Retrieved chunks are not relevant. Retrying retrieval")

        new_query = evaluation["rewritten_query"]

        retry_chunks = search(new_query, k=k)

        logger.info(
            f"Retry retrieval returned {len(retry_chunks)} chunks"
        )

        retry_evaluation = evaluate_retrieval(
            standalone_question,
            retry_chunks,
        )

        retried = True

        if retry_evaluation["relevant"]:
            chunks = retry_chunks
            logger.info("Retry retrieval succeeded")

        else:
            logger.warning(
                "Retry retrieval failed. Falling back to web search"
            )

            # 10. Web Search Fallback
            web_result = answer_from_web(standalone_question)

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

    # 11. Generate answer from retrieved PDF chunks
    text, meta = ask_llm(
        build_prompt(standalone_question, chunks),
        model=GENERATOR_MODEL,
        system=SYSTEM,
        return_meta=True,
    )

    logger.info("Answer generated successfully")

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


# ==============================
# Main
# ==============================

if __name__ == "__main__":

    history = []
    conversation_summary = ""

    logger.info("Adaptive RAG application started")

    while True:

        q = input("\nQuestion (or q to quit): ").strip()

        if q.lower() == "q":
            logger.info("User exited the application")
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

            # Save user question and assistant answer
            history.append({
                "role": "user",
                "content": q,
            })

            history.append({
                "role": "assistant",
                "content": result["answer"],
            })

            logger.debug(
                f"Conversation history now contains {len(history)} messages"
            )

            # Keep only the last 3 question-answer pairs
            if len(history) > 6:

                old_messages = history[:-6]

                try:
                    conversation_summary = update_conversation_summary(
                        conversation_summary,
                        old_messages,
                    )

                    history = history[-6:]

                    logger.info("Old messages summarized and history trimmed")

                except Exception:
                    logger.exception("Failed to update conversation summary")
                    print("Summary update error. Check the log file.")

            # Display current memory
            print("\nHop Type:")
            print(result["hop_type"])

            print("\nCurrent Summary:")
            print(conversation_summary)

            print("\nLast 3 Questions and Answers:")
            print(history)

        except Exception:
            logger.exception("Error while processing question")
            print("An error occurred. Check the log file.")

    logger.info("Adaptive RAG application stopped")
