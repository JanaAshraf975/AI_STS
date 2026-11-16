
import logging
import time

from ddgs import DDGS
from logger_config import log_event


def elapsed_ms(start):
    return round((time.perf_counter() - start) * 1000, 2)


def web_search(query, max_results=5):
    """Search the web using DDGS and return formatted results."""

    start = time.perf_counter()

    log_event(
        logging.INFO,
        "web_search",
        "search_started",
        "Starting web search",
        query_length=len(query),
        max_results=max_results
    )

    try:
        results = DDGS().text(
            query,
            max_results=max_results,
        )

        formatted_results = [
            {
                "title": result.get("title", ""),
                "url": result.get("href", ""),
                "text": result.get("body", ""),
            }
            for result in results
        ]

        log_event(
            logging.INFO,
            "web_search",
            "search_completed",
            "Web search completed successfully",
            results_count=len(formatted_results),
            max_results=max_results,
            result_urls=[
                result["url"]
                for result in formatted_results
            ],
            latency_ms=elapsed_ms(start)
        )

        if not formatted_results:
            log_event(
                logging.WARNING,
                "web_search",
                "no_results",
                "Web search returned no results",
                max_results=max_results
            )

        return formatted_results

    except Exception as e:
        log_event(
            logging.ERROR,
            "web_search",
            "search_failed",
            "Web search failed",
            error_type=type(e).__name__,
            error_message=str(e)[:300],
            latency_ms=elapsed_ms(start)
        )
        raise


# --------------------------------------------------
# Interactive Web Search
# --------------------------------------------------

if __name__ == "__main__":

    print("\n=== Adaptive RAG Web Search ===")
    print("Enter any question to search the internet.")
    print("Type 'q' to quit.\n")

    while True:

        query = input(
            "\nEnter your search query (or 'q' to quit): "
        ).strip()

        if query.lower() == "q":
            print("\nExiting web search...")
            break

        if not query:
            print("Please enter a valid search query.")
            continue

        try:
            results = web_search(
                query,
                max_results=5
            )

            if not results:
                print("\nNo search results found.")
                continue

            print(f"\nFound {len(results)} results:\n")

            for index, result in enumerate(results, start=1):

                print(f"\n--- Result {index} ---")
                print("Title:", result["title"])
                print("URL:", result["url"])
                print("Description:", result["text"])
                print("-" * 50)

        except Exception as e:
            print(
                f"\nWeb search failed: {type(e).__name__}: {e}"
            )
            print("Check logs/error.log for details.")
