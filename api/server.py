import os
import json
import logging
import time
from collections import defaultdict, deque

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from graphs.workflow import run_workflow, stream_workflow
from tools.search_tools import web_search
from tools.browser_tools import fetch_page_content
from tools.llm_tools import get_llm, parse_llm_json, invoke_llm
from tools.logging_config import configure_logging
from memory import store as run_store

load_dotenv()
configure_logging()

logger = logging.getLogger(__name__)

if not os.getenv("GROQ_API_KEY"):
    logger.warning("GROQ_API_KEY is not set — planner/extraction/router calls will fail. See .env.example.")


def require_api_key(x_api_key: str | None = Header(default=None)):
    """No-op unless API_KEY is set in the environment — opt-in protection for public deployments,
    so a stranger with your URL can't spend your Groq/Tavily credits."""
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


# In-memory sliding-window rate limit, keyed by API key (or client IP if no key is set).
# Single-process only — resets on restart and isn't shared across workers/replicas; that's
# an acceptable tradeoff for this project's scale, not meant as a distributed rate limiter.
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
_request_log: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request, x_api_key: str | None = Header(default=None)):
    if RATE_LIMIT_PER_MINUTE <= 0:
        return
    key = x_api_key or (request.client.host if request.client else "unknown")
    now = time.monotonic()
    window = _request_log[key]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded — try again shortly")
    window.append(now)


async def _deliver_webhook(url: str, payload) -> None:
    """Best-effort POST of the final result to a caller-supplied webhook. Never raises —
    a broken webhook shouldn't fail the run that already completed successfully."""
    try:
        body = payload.model_dump() if hasattr(payload, "model_dump") else payload
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(url, json=body)
        logger.info("Webhook delivered to %s (status %d)", url, resp.status_code)
    except Exception as e:
        logger.warning("Webhook delivery to %s failed: %s: %s", url, type(e).__name__, e)

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a tool router. Given user input, decide which tool to use.

Available tools:
- run_agent: for high level goals like 'find jobs', 'research topic', 'collect data', 'find products'
- search_web: for simple searches like 'search for X', 'find articles about Y'
- browse_page: when user gives a specific URL or says 'open/visit/check this site'

Respond ONLY with valid JSON like:
{{"tool": "run_agent", "input": "cleaned up user input"}}

No markdown, no explanation."""),
    ("human", "User input: {input}"),
])


def route_input(user_input: str) -> dict:
    """Ask the LLM which tool to use for a given free-form input. Shared by /auto and /auto/stream."""
    chain = ROUTER_PROMPT | get_llm()
    response = invoke_llm(chain, {"input": user_input})
    try:
        decision = parse_llm_json(response.content)
    except Exception:
        decision = {"tool": "run_agent", "input": user_input}
    return {
        "tool": decision.get("tool", "run_agent"),
        "input": decision.get("input", user_input),
    }

app = FastAPI(
    title="Autonomous Browser Agent API",
    description="AI agent that browses the web and extracts structured data",
    version="1.0.0",
)

_cors_origins = os.getenv("CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("screenshots", exist_ok=True)
app.mount("/screenshots", StaticFiles(directory="screenshots"), name="screenshots")


def _screenshot_url(path: str | None) -> str | None:
    """Map a local screenshots/<run_id>/<file>.png path to its served /screenshots/... URL."""
    if not path:
        return None
    return "/" + path.replace("\\", "/").lstrip("/")


# ─────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────

class GoalRequest(BaseModel):
    goal: str
    capture_screenshots: bool = False
    webhook_url: str | None = None

class SearchRequest(BaseModel):
    query: str
    max_results: int = 5

class BrowseRequest(BaseModel):
    url: str

class JobResult(BaseModel):
    role: str | None
    company: str | None
    location: str | None
    salary: str | None
    apply_url: str | None
    source_url: str | None
    confidence: float | None = None
    screenshot_url: str | None = None

class AgentResponse(BaseModel):
    status: str
    goal: str
    pages_visited: int
    jobs_found: int
    jobs: list[JobResult]
    human_question: str | None = None
    run_id: str | None = None

class AutoRequest(BaseModel):
    input: str

class ClarifyRequest(BaseModel):
    goal: str
    answer: str


def _to_agent_response(result: dict, goal: str) -> AgentResponse:
    """Shared by /run, /run/clarify, and /mcp/run_agent."""
    screenshots = result.get("screenshots", {})
    jobs = [
        JobResult(
            role=job.get("role"),
            company=job.get("company"),
            location=job.get("location"),
            salary=job.get("salary"),
            apply_url=job.get("apply_url"),
            source_url=job.get("source_url"),
            confidence=job.get("confidence"),
            screenshot_url=_screenshot_url(screenshots.get(job.get("source_url"))),
        )
        for job in result.get("extracted_jobs", [])
    ]
    return AgentResponse(
        status=result.get("status", "unknown"),
        goal=goal,
        pages_visited=len(result.get("urls_visited", [])),
        jobs_found=len(jobs),
        jobs=jobs,
        human_question=result.get("human_question"),
        run_id=result.get("run_id"),
    )


@app.post("/auto", dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def auto_route(request: AutoRequest):
    """
    Auto-decides which tool to use based on user input.
    Uses Groq LLM to route the request.
    """
    route = route_input(request.input)
    tool = route["tool"]
    cleaned_input = route["input"]

    logger.info("Auto router: '%s' → %s", request.input, tool)

    # ── call the right tool ──
    if tool == "search_web":
        results = web_search(cleaned_input, 5)
        return {
            "tool_used": "search_web",
            "input": cleaned_input,
            "total": len(results),
            "results": results,
        }

    elif tool == "browse_page":
        content = await fetch_page_content(cleaned_input)
        if not content:
            raise HTTPException(status_code=422, detail=f"Could not fetch {cleaned_input}")
        return {
            "tool_used": "browse_page",
            "url": cleaned_input,
            "content_length": len(content),
            "content": content[:5000],
        }

    else:  # run_agent
        result = await run_workflow(cleaned_input)
        return {"tool_used": "run_agent", **_to_agent_response(result, cleaned_input).model_dump()}


# ─────────────────────────────────────────
# Live Streaming Endpoint
# ─────────────────────────────────────────

NODE_LABELS = {
    "planner":    "Planner: analyzing your goal...",
    "search":     "Search: finding relevant URLs...",
    "browser":    "Browser: visiting pages in parallel...",
    "extraction": "Extraction: pulling out structured data...",
    "validator":  "Validator: checking result quality...",
}

@app.post("/auto/stream", dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def auto_stream(request: AutoRequest):
    """
    Streaming version of /auto.
    Emits SSE events as the agent progresses, then sends a final 'done' event.
    Each event is a JSON object with an 'event' field.
    """
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    async def event_stream():
        # ── Step 1: route ──
        route = route_input(request.input)
        tool = route["tool"]
        cleaned_input = route["input"]

        logger.info("[AutoStream] '%s' → %s", request.input, tool)

        yield f"data: {json.dumps({'event': 'route', 'tool': tool, 'input': cleaned_input})}\n\n"

        # ── Step 2: execute ──
        if tool == "search_web":
            results = web_search(cleaned_input, 5)
            yield f"data: {json.dumps({'event': 'done', 'tool_used': 'search_web', 'input': cleaned_input, 'total': len(results), 'results': results})}\n\n"

        elif tool == "browse_page":
            content = await fetch_page_content(cleaned_input)
            if not content:
                yield f"data: {json.dumps({'event': 'error', 'message': f'Could not fetch {cleaned_input}'})}\n\n"
            else:
                yield f"data: {json.dumps({'event': 'done', 'tool_used': 'browse_page', 'url': cleaned_input, 'content_length': len(content), 'content': content[:5000]})}\n\n"

        else:  # run_agent — stream real node-by-node progress
            last_state = None
            async for node_name, state in stream_workflow(cleaned_input):
                last_state = state
                label = NODE_LABELS.get(node_name, node_name)
                pages = len(state.get("urls_visited", []))
                items = len(state.get("extracted_jobs", []))
                payload = {'event': 'node', 'node': node_name, 'label': label, 'pages': pages, 'items': items, 'status': state.get('status')}
                if node_name == "search":
                    payload['search_query'] = state.get('search_query')
                    payload['attempt'] = state.get('failed_attempts', 0) + 1
                yield f"data: {json.dumps(payload)}\n\n"

            if last_state and last_state.get("status") == "needs_input":
                yield f"data: {json.dumps({'event': 'needs_input', 'question': last_state.get('human_question'), 'goal': cleaned_input})}\n\n"
            elif last_state:
                results = last_state.get("extracted_jobs", [])
                yield f"data: {json.dumps({'event': 'done', 'tool_used': 'run_agent', 'status': last_state.get('status'), 'goal': cleaned_input, 'pages_visited': len(last_state.get('urls_visited', [])), 'jobs_found': len(results), 'jobs': results})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─────────────────────────────────────────
# Core Endpoints
# ─────────────────────────────────────────

@app.get("/")
def health_check():
    return {
        "status": "running",
        "message": "Autonomous Browser Agent is live 🚀",
        "tools": [
            "/run",
            "/run/stream",
            "/run/clarify",
            "/runs",
            "/mcp/search_web",
            "/mcp/browse_page",
            "/mcp/run_agent",
        ]
    }


@app.post("/run", response_model=AgentResponse, dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def run_agent(request: GoalRequest):
    """
    Run the full agent workflow for a given goal.
    Returns structured job results — or, if the goal was too vague, a status of
    'needs_input' and a `human_question` to answer via POST /run/clarify.
    """
    if not request.goal.strip():
        raise HTTPException(status_code=400, detail="Goal cannot be empty")

    logger.info("New request: %s", request.goal)

    if request.webhook_url and not request.webhook_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="webhook_url must be an http:// or https:// URL")

    try:
        result = await run_workflow(request.goal, capture_screenshots=request.capture_screenshots)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    response = _to_agent_response(result, request.goal)
    if request.webhook_url:
        await _deliver_webhook(request.webhook_url, response)
    return response


@app.post("/run/clarify", response_model=AgentResponse, dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def run_clarify(request: ClarifyRequest):
    """
    Answer a clarifying question the planner asked (see AgentResponse.human_question) and
    re-run the workflow with that detail folded into the goal.

    Note: this is a pragmatic, stateless resume (append the answer to the goal and start a
    fresh run) — not a true mid-graph pause/resume, which would need a LangGraph checkpointer
    keyed by thread id. It's enough to unblock a goal that was too vague on the first pass.
    """
    if not request.goal.strip() or not request.answer.strip():
        raise HTTPException(status_code=400, detail="goal and answer are both required")

    combined_goal = f"{request.goal}\n\nAdditional detail from user: {request.answer}"
    try:
        result = await run_workflow(combined_goal)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    return _to_agent_response(result, request.goal)


@app.get("/runs", dependencies=[Depends(require_api_key)])
def list_run_history(limit: int = 20, offset: int = 0, status: str | None = None):
    """Recent workflow runs (goal, status, result counts) — persisted to SQLite.
    Optionally filter by `status` (e.g. 'done', 'error', 'needs_input') and page with
    `limit`/`offset`."""
    return {
        "runs": run_store.list_runs(limit=limit, status=status, offset=offset),
        "total": run_store.count_runs(status=status),
        "limit": limit,
        "offset": offset,
    }


@app.get("/runs/{run_id}", dependencies=[Depends(require_api_key)])
def get_run_detail(run_id: str):
    """Full detail (including extracted jobs) for one past run."""
    run = run_store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.post("/run/stream", dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def run_agent_stream(request: GoalRequest):
    """
    Run the agent and stream real node-by-node progress live (SSE, plain-text lines).
    """
    if not request.goal.strip():
        raise HTTPException(status_code=400, detail="Goal cannot be empty")
    if request.webhook_url and not request.webhook_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="webhook_url must be an http:// or https:// URL")

    async def event_stream():
        icons = {"planner": "🧠", "search": "🔍", "browser": "🌐", "extraction": "📋", "validator": "✅"}
        try:
            last_state = None
            async for node_name, state in stream_workflow(request.goal, capture_screenshots=request.capture_screenshots):
                last_state = state
                label = NODE_LABELS.get(node_name, node_name)
                if node_name == "search":
                    attempt = state.get("failed_attempts", 0) + 1
                    label = f"{label} (attempt #{attempt}: '{state.get('search_query')}')"
                yield f"data: {icons.get(node_name, '⚙️')} {label}\n\n"

            last_state = last_state or {}
            if last_state.get("status") == "needs_input":
                yield f"data: 🤔 {last_state.get('human_question')}\n\n"
                yield "data: Answer via POST /run/clarify with {\"goal\": ..., \"answer\": ...}\n\n"
            else:
                jobs = last_state.get("extracted_jobs", [])
                yield f"data: ✅ Done! Found {len(jobs)} results.\n\n"
                for job in jobs:
                    yield f"data: 💼 {job.get('role')} @ {job.get('company')} — {job.get('location')}\n\n"
                if request.webhook_url:
                    await _deliver_webhook(request.webhook_url, _to_agent_response(last_state, request.goal))
        except Exception as e:
            yield f"data: ❌ Error: {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─────────────────────────────────────────
# MCP Tool Endpoints
# ─────────────────────────────────────────

@app.get("/mcp/tools")
def list_mcp_tools():
    """List all available MCP tools."""
    return {
        "tools": [
            {
                "name": "search_web",
                "description": "Search the web for any query using Tavily",
                "endpoint": "POST /mcp/search_web",
                "input": {"query": "string", "max_results": "int (default 5)"},
            },
            {
                "name": "browse_page",
                "description": "Visit a URL and return full page text",
                "endpoint": "POST /mcp/browse_page",
                "input": {"url": "string"},
            },
            {
                "name": "run_agent",
                "description": "Run full autonomous agent workflow for a goal",
                "endpoint": "POST /mcp/run_agent",
                "input": {"goal": "string"},
            },
        ]
    }


@app.post("/mcp/search_web", dependencies=[Depends(require_api_key), Depends(rate_limit)])
def mcp_search_web(request: SearchRequest):
    """
    MCP Tool — search_web
    Search the web using Tavily and return URLs + snippets.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info("MCP search_web: %s", request.query)
    results = web_search(request.query, request.max_results)

    return {
        "tool": "search_web",
        "query": request.query,
        "total": len(results),
        "results": results,
    }


@app.post("/mcp/browse_page", dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def mcp_browse_page(request: BrowseRequest):
    """
    MCP Tool — browse_page
    Visit a URL with a real browser and return page text.
    """
    if not request.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    logger.info("MCP browse_page: %s", request.url)
    content = await fetch_page_content(request.url)

    if not content:
        raise HTTPException(
            status_code=422,
            detail=f"Could not fetch content from {request.url}"
        )

    return {
        "tool": "browse_page",
        "url": request.url,
        "content_length": len(content),
        "content": content[:5000],
    }


@app.post("/mcp/run_agent", response_model=AgentResponse, dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def mcp_run_agent(request: GoalRequest):
    """
    MCP Tool — run_agent
    Run the full autonomous agent workflow for a high level goal.
    """
    if not request.goal.strip():
        raise HTTPException(status_code=400, detail="Goal cannot be empty")

    logger.info("MCP run_agent: %s", request.goal)

    try:
        result = await run_workflow(request.goal)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    return _to_agent_response(result, request.goal)