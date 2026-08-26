import asyncio
import logging
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from tools.search_tools import web_search
from tools.browser_tools import fetch_page_content
from tools.logging_config import configure_logging
from graphs.workflow import run_workflow

load_dotenv()
configure_logging()

logger = logging.getLogger(__name__)

# create MCP server
app = Server("browser-agent")


# ─────────────────────────────────────────
# Tool 1 — search_web
# ─────────────────────────────────────────
@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_web",
            description="Search the web for any query using Tavily. Returns list of URLs and snippets.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="browse_page",
            description="Visit a URL using a real browser and return the page text content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to visit",
                    },
                },
                "required": ["url"],
            },
        ),
        types.Tool(
            name="run_agent",
            description="""Run the full autonomous agent workflow.
Give it a high level goal like 'Find remote AI internships'.
It will search, browse, extract and return structured results.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "The high level goal for the agent",
                    },
                },
                "required": ["goal"],
            },
        ),
    ]


# ─────────────────────────────────────────
# Tool execution
# ─────────────────────────────────────────
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    # ── search_web ──
    if name == "search_web":
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)

        logger.info("MCP search_web: %s", query)
        results = web_search(query, max_results)

        output = f"Search results for: '{query}'\n\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. {r['title']}\n"
            output += f"   URL: {r['url']}\n"
            output += f"   {r['snippet']}\n\n"

        return [types.TextContent(type="text", text=output)]

    # ── browse_page ──
    elif name == "browse_page":
        url = arguments.get("url", "")

        logger.info("MCP browse_page: %s", url)
        content = await fetch_page_content(url)

        if not content:
            return [types.TextContent(
                type="text",
                text=f"❌ Could not fetch content from {url}"
            )]

        return [types.TextContent(
            type="text",
            text=f"Page content from {url}:\n\n{content[:5000]}"
        )]

    # ── run_agent ──
    elif name == "run_agent":
        goal = arguments.get("goal", "")

        logger.info("MCP run_agent: %s", goal)
        result = await run_workflow(goal)

        jobs = result.get("extracted_jobs", [])
        status = result.get("status", "unknown")
        pages_visited = len(result.get("urls_visited", []))

        output = f"Agent completed with status: {status}\n"
        output += f"Pages visited: {pages_visited}\n"
        output += f"Jobs found: {len(jobs)}\n\n"

        for i, job in enumerate(jobs, 1):
            output += f"#{i}\n"
            output += f"  Role     : {job.get('role')}\n"
            output += f"  Company  : {job.get('company')}\n"
            output += f"  Location : {job.get('location')}\n"
            output += f"  Salary   : {job.get('salary')}\n"
            output += f"  Apply    : {job.get('apply_url')}\n"
            output += f"  Source   : {job.get('source_url')}\n\n"

        return [types.TextContent(type="text", text=output)]

    else:
        return [types.TextContent(
            type="text",
            text=f"❌ Unknown tool: {name}"
        )]


# ─────────────────────────────────────────
# Run the server
# ─────────────────────────────────────────
async def main():
    logger.info("Browser Agent MCP Server starting...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())