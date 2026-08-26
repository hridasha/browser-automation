import asyncio
import logging

from tools.browser_tools import fetch_pages
from memory.state_manager import AgentState

logger = logging.getLogger(__name__)

# Bound on how many extra pages a crawl can pull in beyond the original search results,
# so a goal that sets crawl_depth > 1 can't turn into an unbounded crawl.
MAX_CRAWLED_PAGES_MULTIPLIER = 3


async def browser_node_async(state: AgentState) -> AgentState:
    search_results = state.get("search_results", [])
    urls_visited = list(state.get("urls_visited", []))
    screenshots = dict(state.get("screenshots", {}))
    crawl_depth = max(1, state.get("crawl_depth", 1))
    capture_screenshots = state.get("capture_screenshots", False)
    max_results = state.get("max_results", 5)
    screenshot_dir = f"screenshots/{state['run_id']}" if capture_screenshots else None

    to_fetch = [r for r in search_results if r.get("url") and r["url"] not in urls_visited]
    skipped = len(search_results) - len(to_fetch)

    logger.info("---- Browser START (parallel, shared browser) ----")
    logger.info("Total search results : %d", len(search_results))
    logger.debug("Already visited      : %d skipped", skipped)
    logger.info("Fetching in parallel : %d URLs (crawl_depth=%d)", len(to_fetch), crawl_depth)

    pages = []
    level_meta = {r["url"]: r for r in to_fetch}  # url -> originating search result (title/snippet)
    level_urls = [r["url"] for r in to_fetch]
    max_total = max_results * MAX_CRAWLED_PAGES_MULTIPLIER

    for depth in range(crawl_depth):
        if not level_urls:
            break

        collect_links = depth < crawl_depth - 1  # no need to extract links from the last level
        fetched = await fetch_pages(level_urls, collect_links=collect_links, screenshot_dir=screenshot_dir)

        next_level_urls = []
        for url in level_urls:
            result = fetched.get(url, {})
            content = result.get("content")
            if content:
                meta = level_meta.get(url, {})
                pages.append({
                    "url": url,
                    "title": meta.get("title", ""),
                    "snippet": meta.get("snippet", ""),
                    "content": content,
                })
                urls_visited.append(url)
                if result.get("screenshot"):
                    screenshots[url] = result["screenshot"]
                for link in result.get("links", []):
                    if link not in urls_visited and link not in next_level_urls and len(urls_visited) + len(next_level_urls) < max_total:
                        next_level_urls.append(link)
            else:
                logger.warning("No content: %s", url)

        if next_level_urls:
            logger.info("Crawl depth %d found %d new same-domain link(s) to follow", depth + 1, len(next_level_urls))
        level_urls = next_level_urls
        level_meta = {}

    logger.info("Successfully fetched : %d pages total", len(pages))
    logger.debug("Total visited so far : %d", len(urls_visited))
    logger.info("---- Browser DONE ----")

    return {
        **state,
        "urls_visited": urls_visited,
        "pages": pages,
        "screenshots": screenshots,
        "current_task": "extracting",
        "status": "extracting",
    }


def browser_node(state: AgentState) -> AgentState:
    return asyncio.run(browser_node_async(state))
