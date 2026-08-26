import logging
import os
from typing import List, Dict
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

logger = logging.getLogger(__name__)

SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "tavily").lower()


def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """Provider-agnostic web search. Selected via SEARCH_PROVIDER (default 'tavily').
    To add a provider: write a `_<name>_search(query, max_results) -> List[Dict]` function
    returning {"url", "title", "snippet"} dicts, and register it below."""
    if SEARCH_PROVIDER == "tavily":
        return _tavily_search(query, max_results)
    raise ValueError(f"Unknown SEARCH_PROVIDER '{SEARCH_PROVIDER}' (expected: tavily)")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _tavily_search_with_retry(client, query: str, max_results: int):
    return client.search(query=query, max_results=max_results)


def _tavily_search(query: str, max_results: int) -> List[Dict]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set — falling back to mock results")
        return _mock_search(query, max_results)

    logger.info("Tavily request: query='%s', max_results=%d", query, max_results)
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = _tavily_search_with_retry(client, query, max_results)
        raw_count = len(response.get("results", []))
        logger.debug("Tavily returned %d raw results", raw_count)
        results = []
        for r in response.get("results", []):
            results.append({
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "snippet": r.get("content", ""),
            })
        logger.info("Tavily parsed %d results", len(results))
        return results
    except Exception as e:
        # A real API call failed (bad key, quota, network) — unlike the "no key configured"
        # case above, silently substituting mock example.com URLs here would let the retry
        # loop burn all 3 attempts fetching pages that can never resolve, hiding the real
        # cause. Return no results instead, so the caller's existing retry/abort logic
        # (based on empty results) kicks in immediately with the error visible in logs.
        logger.error("Tavily error: %s: %s — returning no results", type(e).__name__, e)
        return []


def _mock_search(query: str, max_results: int) -> List[Dict]:
    return [
        {
            "url": f"https://example-jobs.com/job-{i}",
            "title": f"AI Intern Role #{i}",
            "snippet": f"Mock result {i} for query: {query}",
        }
        for i in range(1, max_results + 1)
    ]
