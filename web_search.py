
from ddgs import DDGS


def web_search(query, max_results=5):
    results = DDGS().text(
        query,
        max_results=max_results,
    )

    return [
        {
            "title": result.get("title", ""),
            "url": result.get("href", ""),
            "text": result.get("body", ""),
        }
        for result in results
    ]

if __name__ == "__main__":
    results = web_search("What is Adaptive RAG?")

    for result in results:
        print(result["title"])
        print(result["url"])
        print(result["text"])
        print("-" * 50)
