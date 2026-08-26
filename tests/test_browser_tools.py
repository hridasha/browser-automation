import asyncio

from playwright.async_api import async_playwright

from tools.browser_tools import extract_links, _collapse_duplicate_lines


def test_extract_links_filters_to_same_domain_and_dedupes():
    html = """
    <html><body>
      <a href="https://example.com/jobs/1">Job 1</a>
      <a href="https://example.com/jobs/2">Job 2</a>
      <a href="https://example.com/jobs/1#section">Job 1 dup with fragment</a>
      <a href="https://www.example.com/jobs/3">Job 3 (www variant)</a>
      <a href="https://other-site.com/jobs/4">Off-domain</a>
      <a href="mailto:someone@example.com">Not a link</a>
    </body></html>
    """

    async def run():
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                await page.set_content(html)
                return await extract_links(page, "https://example.com/careers")
            finally:
                await browser.close()

    links = asyncio.run(run())

    assert set(links) == {
        "https://example.com/jobs/1",
        "https://example.com/jobs/2",
        "https://www.example.com/jobs/3",
    }


def test_collapse_duplicate_lines_removes_immediate_repeat():
    text = (
        "Machine Learning Intern\n"
        "Machine Learning Intern\n"
        "Acme Corp\n"
        "Remote\n"
    )
    assert _collapse_duplicate_lines(text) == (
        "Machine Learning Intern\n"
        "Acme Corp\n"
        "Remote\n"
    )


def test_collapse_duplicate_lines_leaves_non_adjacent_repeats():
    text = "Acme Corp\nRemote\nAcme Corp\n"
    assert _collapse_duplicate_lines(text) == text


def test_collapse_duplicate_lines_leaves_blank_lines_untouched():
    text = "Title\n\n\nCompany\n"
    assert _collapse_duplicate_lines(text) == text
