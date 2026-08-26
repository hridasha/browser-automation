import asyncio
import logging
import re
from typing import Optional
from urllib.parse import urlparse
from playwright.async_api import async_playwright, Browser, Page

logger = logging.getLogger(__name__)


def _registrable_domain(url: str) -> str:
    """Best-effort 'same site' comparison: strip a leading www. and use the bare host,
    e.g. https://www.example.com/x and https://jobs.example.com both -> not equal unless
    hosts match exactly after stripping www. Good enough for same-domain link filtering
    without pulling in a public-suffix-list dependency."""
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


async def extract_links(page: Page, base_url: str) -> list[str]:
    """Same-domain links found on an already-loaded page, deduped and resolved to absolute URLs."""
    try:
        hrefs = await page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
    except Exception as e:
        logger.debug("Link extraction failed for %s: %s: %s", base_url, type(e).__name__, e)
        return []

    base_domain = _registrable_domain(base_url)
    seen = set()
    links = []
    for href in hrefs:
        if not href or not href.startswith(("http://", "https://")):
            continue
        if _registrable_domain(href) != base_domain:
            continue
        clean = href.split("#")[0]
        if clean in seen or clean == base_url:
            continue
        seen.add(clean)
        links.append(clean)
    return links


def _collapse_duplicate_lines(text: str) -> str:
    """Collapse a line that immediately repeats the line before it. Many sites (LinkedIn's
    job cards, for one) render a visible heading plus a visually-hidden duplicate of the same
    text for screen readers, and innerText captures both back-to-back. Left in, this clutter
    roughly doubles the length of every listing entry and makes it harder for the extraction
    LLM to correctly pair each title with its company/location across a long list."""
    collapsed = []
    prev_stripped = None
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and stripped == prev_stripped:
            continue
        collapsed.append(line)
        prev_stripped = stripped
    return "\n".join(collapsed)


async def _extract_from_page(
    browser: Browser, url: str, timeout: int, collect_links: bool = False, screenshot_path: Optional[str] = None
) -> tuple[Optional[str], list[str]]:
    page = await browser.new_page()
    try:
        await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
        content = await page.evaluate("() => document.body.innerText")
        content = _collapse_duplicate_lines(content)
        links = await extract_links(page, url) if collect_links else []
        if screenshot_path:
            try:
                await page.screenshot(path=screenshot_path)
            except Exception as e:
                logger.warning("Screenshot failed for %s: %s: %s", url, type(e).__name__, e)
        return content[:8000], links
    finally:
        await page.close()


async def fetch_page_content(url: str, timeout: int = 15000) -> Optional[str]:
    """Fetch a single page's text content. Launches its own browser — use fetch_pages()
    instead when fetching multiple URLs so they can share one browser instance."""
    logger.debug("Launching Chromium for: %s", url)
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                content, _links = await _extract_from_page(browser, url, timeout)
            finally:
                await browser.close()
            logger.debug("Extracted %d chars from %s", len(content) if content else 0, url)
            return content
    except Exception as e:
        logger.error("Error on %s: %s: %s", url, type(e).__name__, e)
        return None


def _slug_for(url: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", url).strip("-")[:120]


async def fetch_pages(
    urls: list[str],
    timeout: int = 15000,
    collect_links: bool = False,
    screenshot_dir: Optional[str] = None,
) -> dict[str, dict]:
    """Fetch multiple URLs concurrently, sharing a single browser instance
    (one page/tab per URL) instead of spawning a Chromium process per URL.

    Returns {url: {"content": str|None, "links": list[str], "screenshot": str|None}}.
    - `collect_links`: also return same-domain links found on each page (for crawling).
    - `screenshot_dir`: if set, save a screenshot of each page there and report its path.
    """
    results: dict[str, dict] = {}
    if not urls:
        return results

    if screenshot_dir:
        import os
        os.makedirs(screenshot_dir, exist_ok=True)

    logger.info("Launching shared Chromium for %d URL(s)", len(urls))
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            async def fetch_one(url: str):
                screenshot_path = f"{screenshot_dir}/{_slug_for(url)}.png" if screenshot_dir else None
                try:
                    content, links = await _extract_from_page(
                        browser, url, timeout, collect_links=collect_links, screenshot_path=screenshot_path
                    )
                    logger.debug("✓ %s (%d chars, %d links)", url, len(content) if content else 0, len(links))
                    return url, {
                        "content": content,
                        "links": links,
                        "screenshot": screenshot_path if content else None,
                    }
                except Exception as e:
                    logger.error("Error on %s: %s: %s", url, type(e).__name__, e)
                    return url, {"content": None, "links": [], "screenshot": None}

            fetched = await asyncio.gather(*[fetch_one(u) for u in urls])
            results = dict(fetched)
        finally:
            await browser.close()

    return results
