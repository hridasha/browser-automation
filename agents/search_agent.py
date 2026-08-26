import logging

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from tools.search_tools import web_search
from memory.state_manager import AgentState
from tools.llm_tools import get_llm, invoke_llm

load_dotenv()

logger = logging.getLogger(__name__)

RETRY_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a search query optimizer.
The previous search query did not return useful results.
Generate a DIFFERENT, more specific search query for the same goal.
Try a different angle — use different keywords, be more specific, or target a different site type.
Respond with ONLY the new query string. No explanation, no quotes."""),
    ("human", "Goal: {goal}\nFailed query: {query}\nAttempt: {attempt}"),
])


def search_node(state: AgentState) -> AgentState:
    base_query = state.get("search_query", state["user_goal"])
    failed_attempts = state.get("failed_attempts", 0)
    max_results = state.get("max_results", 5)

    logger.info("---- Search START ----")

    if failed_attempts > 0:
        logger.info("Retry #%d — asking LLM for a better query...", failed_attempts)
        chain = RETRY_QUERY_PROMPT | get_llm()
        response = invoke_llm(chain, {
            "goal": state["user_goal"],
            "query": base_query,
            "attempt": failed_attempts + 1,
        })
        query = response.content.strip().strip('"').strip("'")
        logger.info("Old query : '%s'", base_query)
        logger.info("New query : '%s'", query)
    else:
        query = base_query
        logger.info("Query : '%s'", query)

    logger.debug("Max results: %d", max_results)

    results = web_search(query, max_results)

    logger.info("Got %d results", len(results))
    for i, r in enumerate(results, 1):
        logger.debug("  %d. %s — %s", i, r.get("title", "(no title)"), r["url"])

    logger.info("---- Search DONE ----")

    return {
        **state,
        "search_query": query,
        "search_results": results,
        "current_task": "browsing",
        "status": "browsing",
    }
