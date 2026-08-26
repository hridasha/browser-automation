import logging
import uuid

from langgraph.graph import StateGraph, END
from memory.state_manager import AgentState
from memory import store
from agents.planner import planner_node
from agents.search_agent import search_node
from agents.browser_agent import browser_node
from agents.extraction_agent import extraction_node
from agents.validator import validator_node

logger = logging.getLogger(__name__)


def should_retry_or_end(state: AgentState) -> str:
    status = state.get("status")
    failed_attempts = state.get("failed_attempts", 0)

    logger.debug("[Router] status='%s' | failed_attempts=%d", status, failed_attempts)

    if status == "done":
        logger.debug("[Router] → END (success)")
        return "end"
    elif status == "error" or failed_attempts >= 3:
        logger.debug("[Router] → END (abort after %d failed attempts)", failed_attempts)
        return "end"
    else:
        logger.debug("[Router] → back to SEARCH for retry #%d", failed_attempts + 1)
        return "retry"


def route_after_planner(state: AgentState) -> str:
    """If the planner couldn't form a plan and needs clarification from the user,
    stop here instead of searching with a bad/empty query."""
    if state.get("status") == "needs_input":
        logger.debug("[Router] → END (planner needs human clarification)")
        return "end"
    return "continue"


def _make_initial_state(user_goal: str, capture_screenshots: bool = False) -> AgentState:
    return {
        "run_id": str(uuid.uuid4()),
        "user_goal": user_goal,
        "current_task": "planning",
        "urls_visited": [],
        "extracted_jobs": [],
        "failed_attempts": 0,
        "status": "planning",
        "search_results": [],
        "reflection_notes": None,
        "human_question": None,
        "pages": [],
        "search_query": "",
        "extraction_fields": [],
        "max_results": 5,
        "crawl_depth": 1,
        "capture_screenshots": capture_screenshots,
        "screenshots": {},
    }


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("search", search_node)
    graph.add_node("browser", browser_node)
    graph.add_node("extraction", extraction_node)
    graph.add_node("validator", validator_node)
    graph.set_entry_point("planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"end": END, "continue": "search"},
    )
    graph.add_edge("search", "browser")
    graph.add_edge("browser", "extraction")
    graph.add_edge("extraction", "validator")
    graph.add_conditional_edges(
        "validator",
        should_retry_or_end,
        {"end": END, "retry": "search"},
    )
    return graph.compile()


async def stream_workflow(user_goal: str, capture_screenshots: bool = False):
    """Async generator — yields (node_name, state) after each node completes.
    Persists the final state to run history once the graph finishes."""
    app = _build_graph()
    initial_state = _make_initial_state(user_goal, capture_screenshots=capture_screenshots)
    last_state = initial_state
    async for chunk in app.astream(initial_state):
        for node_name, state in chunk.items():
            last_state = state
            yield node_name, state
    store.save_run(last_state)


async def run_workflow(user_goal: str, capture_screenshots: bool = False) -> AgentState:
    logger.info("=" * 60)
    logger.info("Building LangGraph for goal: '%s'", user_goal)
    logger.info("Graph: planner → search → browser → extraction → validator")
    logger.info("Conditional edges: planner → (end-if-needs-input | search); validator → (end | retry→search)")

    app = _build_graph()
    initial_state = _make_initial_state(user_goal, capture_screenshots=capture_screenshots)

    logger.debug("Invoking graph (run_id=%s)...", initial_state["run_id"])
    result = await app.ainvoke(initial_state)
    logger.info("Finished — status: %s | results: %d", result.get("status"), len(result.get("extracted_jobs", [])))
    logger.info("=" * 60)

    store.save_run(result)
    return result
