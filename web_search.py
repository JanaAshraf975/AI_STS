
from ddgs import DDGS
from ddgs.exceptions import DDGSException

def web_search(query, max_results=5):
    try:
        results = DDGS().text(
            query,
            max_results=max_results,
        )
        return results or []

    except DDGSException as e:
        print(f"Web search failed: {e}")
        return []

    except Exception as e:
        print(f"Unexpected web search error: {e}")
        return []
