import logging

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from memory.state_manager import AgentState
from tools.llm_tools import get_llm, parse_llm_json, invoke_llm

load_dotenv()

logger = logging.getLogger(__name__)

llm = get_llm()

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an autonomous planning agent.
Given a user goal, first decide whether it is specific enough to act on.

If the goal is too vague or ambiguous to search effectively (e.g. a single word with no
clear domain, or missing a critical detail like topic/location/role), respond with ONLY:
{{"needs_clarification": true, "question": "<one short, specific question for the user>"}}
Only ask when truly necessary — prefer a reasonable assumption over asking. Most goals do
NOT need clarification.

Otherwise, output a concise JSON execution plan with:
- search_query: best web search query to find relevant pages
- extraction_fields: list of fields to extract from each page
- filter_criteria: what makes a result relevant
- max_results: how many results to collect (max 10)
- crawl_depth: how many link-hops to follow from each search result (1 = just the search
  result pages themselves, the default). Only set this above 1 (max 3) if the goal implies
  visiting a listing/index page that links out to individual detail pages — e.g. a careers
  page listing many openings, or a directory page — where following links is needed to reach
  the actual data.

Respond ONLY with valid JSON. No markdown, no explanation."""),
    ("human", "User goal: {goal}"),
])


def planner_node(state: AgentState) -> AgentState:
    logger.info("---- Planner START ----")
    logger.info("Goal received: '%s'", state["user_goal"])
    logger.debug("Calling LLM to generate execution plan...")

    chain = PLANNER_PROMPT | llm
    response = invoke_llm(chain, {"goal": state["user_goal"]})
    logger.debug("Raw LLM response: %s", response.content.strip())

    try:
        plan = parse_llm_json(response.content)
        logger.debug("JSON parsed successfully")
    except Exception as e:
        logger.warning("JSON parse failed (%s), using fallback plan", e)
        plan = {
            "search_query": state["user_goal"],
            "extraction_fields": ["company", "role", "location", "salary", "apply_url"],
            "filter_criteria": "relevant to user goal",
            "max_results": 5,
        }

    if plan.get("needs_clarification"):
        question = plan.get("question") or "Could you clarify your goal?"
        logger.info("Needs clarification: %s", question)
        logger.info("---- Planner DONE (awaiting human) ----")
        return {
            **state,
            "current_task": "awaiting_human",
            "status": "needs_input",
            "human_question": question,
        }

    logger.info("search_query     : %s", plan.get("search_query"))
    logger.info("extraction_fields: %s", plan.get("extraction_fields"))
    logger.info("max_results      : %s", plan.get("max_results"))
    logger.info("crawl_depth      : %s", plan.get("crawl_depth", 1))
    logger.info("---- Planner DONE ----")

    try:
        crawl_depth = max(1, min(3, int(plan.get("crawl_depth", 1))))
    except (TypeError, ValueError):
        crawl_depth = 1

    return {
        **state,
        "current_task": "searching",
        "status": "searching",
        "human_question": None,
        "search_query": plan.get("search_query", state["user_goal"]),
        "extraction_fields": plan.get("extraction_fields", ["company", "role", "location"]),
        "max_results": plan.get("max_results", 5),
        "crawl_depth": crawl_depth,
    }
