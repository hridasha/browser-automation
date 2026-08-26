# Browser Automation Agent

An autonomous multi-agent workflow, built on [LangGraph](https://github.com/langchain-ai/langgraph), that
takes a plain-English goal (e.g. *"find remote AI internships"*), searches the web, visits the resulting
pages with a real headless browser, and extracts structured data from them using an LLM.

```
planner → search → browser → extraction → validator ──done
             ↑                                 │
             └──────────── retry (≤3x) ────────┘
```

- **Planner** — turns the user's goal into a search query + list of fields to extract
- **Search** — queries [Tavily](https://tavily.com) for candidate URLs
- **Browser** — visits every URL in parallel with [Playwright](https://playwright.dev) (one shared
  Chromium instance, one tab per page)
- **Extraction** — asks an LLM ([Groq](https://groq.com) / Llama 3.3) to pull structured fields out of
  each page's text
- **Validator** — drops sparse/low-quality results; if nothing survives, loops back to search with a
  reformulated query (up to 3 attempts) before giving up

The workflow is exposed three ways:

| Interface | Where |
|---|---|
| CLI | `main.py` |
| REST API + SSE streaming + a small dashboard | `api/server.py`, `frontend/` |
| MCP server (stdio) | `mcp_server/server.py` |

## Setup

Requires Python ≥3.11 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run playwright install chromium

cp .env.example .env
# then fill in GROQ_API_KEY and TAVILY_API_KEY in .env
```

- `GROQ_API_KEY` is required — every agent node calls Groq.
- `TAVILY_API_KEY` is optional — without it, search falls back to mock results (useful for local
  testing without spending API credits).

## Running it

**CLI:**
```bash
uv run python main.py
```

**API + dashboard:**
```bash
uv run uvicorn api.server:app --reload
```
Then open `frontend/index.html` in a browser (it talks to `http://localhost:8000` by default).

**MCP server** (for use with an MCP-compatible client):
```bash
uv run python -m mcp_server.server
```

## Tests

```bash
uv run pytest
```

These are fast, offline unit tests (validator logic, JSON parsing). Scripts that hit real, billed APIs
(Groq, Tavily) or launch a real browser live under `scripts/manual_checks/` and are run individually by
hand — see that folder's README.

## Project layout

```
agents/       LangGraph node implementations (planner, search, browser, extraction, validator)
graphs/       Graph wiring (graphs/workflow.py) and the state machine's retry logic
tools/        Shared, testable I/O: search_tools, browser_tools, llm_tools
memory/       The AgentState TypedDict passed between graph nodes
api/          FastAPI app: REST + SSE endpoints
mcp_server/   Stdio MCP server exposing the same tools to MCP clients
frontend/     Static HTML/JS/CSS dashboard for the API
tests/        pytest unit tests (no network calls)
scripts/      One-off manual scripts (not part of CI)
```

## Known limitations

- `memory/state_manager.py` only defines the in-graph state shape — there's no persistence between runs.
- The `human_question` field on `AgentState` is defined but never populated; there's no human-in-the-loop
  step yet.
- CORS defaults to `*`; set `CORS_ORIGINS` in `.env` before deploying the API publicly.
