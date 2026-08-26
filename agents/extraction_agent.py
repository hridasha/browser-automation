import logging
import re

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from memory.state_manager import AgentState
from tools.llm_tools import get_llm, parse_llm_json, invoke_llm

load_dotenv()

logger = logging.getLogger(__name__)

# Different models/runs drift from the exact field names the prompt asks for despite
# being told not to (seen in practice: "position_title" instead of "role", "link" instead
# of "apply_url"). Normalize the common variants so downstream code and the UI, which key
# off canonical names, still find the data.
FIELD_ALIASES = {
    "role": ("job_title", "job title", "title", "position", "position_title", "job_role", "job_name", "internship_title", "internship title"),
    "company": ("company_name", "employer", "organization", "org"),
    "apply_url": ("link", "url", "apply_link", "application_link", "job_url", "job_link", "listing_url", "application_url"),
    "salary": ("compensation", "pay", "salary_range", "pay_range"),
    "job_description": ("job_description", "job description", "description"),
}


def _normalize_key(key: str) -> str:
    """Lowercase and snake_case a field name so a model's own phrasing of a planner-chosen
    field (e.g. planner asks for "Job Title" / "Application Link") still matches known
    canonical/alias keys regardless of exact casing or spacing."""
    return re.sub(r"[\s\-]+", "_", str(key).strip().lower())


# FIELD_ALIASES keyed by normalized alias -> canonical name, built once at import time.
_ALIAS_TO_CANONICAL = {
    _normalize_key(alias): canonical
    for canonical, aliases in FIELD_ALIASES.items()
    for alias in aliases
}


def _canonicalize_keys(data: dict) -> dict:
    """Rename every key to its canonical field name (case/spacing-insensitive). If two
    differently-named keys map to the same canonical name (e.g. a null "company" alongside
    a filled "Company Name"), keep whichever has a real value."""
    result = {}
    for raw_key, value in data.items():
        canonical = _ALIAS_TO_CANONICAL.get(_normalize_key(raw_key), _normalize_key(raw_key))
        if canonical not in result or result[canonical] in (None, "", "null", "N/A"):
            result[canonical] = value
    return result

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a data extraction agent.
Extract structured information from the page content.
Fields to extract: {fields}

Rules:
- Always use the exact field names listed above — do NOT rename them.
- Always use "role" for the job title — never "job_title" or "job title".
- If the page contains multiple jobs, return a JSON array of objects.
- If only one job, return a single JSON object.
- If a field is not found, set it to null.
- Every object must also include a "confidence" field: a number from 0.0 to 1.0 representing
  how confident you are that the extracted values are accurate and complete for that item.
  Use a low value if the page content was ambiguous, boilerplate, or mostly unrelated to the
  requested fields.
- Respond ONLY with valid JSON. No markdown, no explanation."""),
    ("human", "Page URL: {url}\n\nPage Content:\n{content}"),
])


def _normalize_confidence(value) -> float:
    """Coerce a model-reported confidence into [0, 1], defaulting to 0.5 (neutral)
    when missing or not a usable number."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, score))


def extraction_node(state: AgentState) -> AgentState:
    pages = state.get("pages", [])
    fields = state.get("extraction_fields", ["company", "role", "location", "salary", "apply_url"])
    extracted_jobs = list(state.get("extracted_jobs", []))

    logger.info("---- Extraction START ----")
    logger.info("Pages to process  : %d", len(pages))
    logger.debug("Fields to extract : %s", fields)

    for i, page in enumerate(pages, 1):
        url = page.get("url", "unknown")
        content_len = len(page.get("content", ""))
        logger.debug("[%d/%d] Processing: %s (%d chars, sending first 4000)", i, len(pages), url, content_len)

        try:
            chain = EXTRACTION_PROMPT | get_llm()
            response = invoke_llm(chain, {
                "fields": ", ".join(fields),
                "url": url,
                "content": page.get("content", "")[:4000],
            })

            parsed = parse_llm_json(response.content)

            # LLM may return a list (multiple jobs) or a single dict
            items = parsed if isinstance(parsed, list) else [parsed]
            logger.debug("LLM returned %d item(s) for %s", len(items), url)

            for data in items:
                data = _canonicalize_keys(data)
                data["source_url"] = url
                data["confidence"] = _normalize_confidence(data.get("confidence"))
                extracted_jobs.append(data)

                logger.info("Extracted: role=%s | company=%s | location=%s | confidence=%.2f",
                            data.get("role"), data.get("company"), data.get("location"), data["confidence"])

        except Exception as e:
            logger.error("Extraction failed for %s — %s: %s", url, type(e).__name__, e)

    logger.info("Total extracted records : %d", len(extracted_jobs))
    logger.info("---- Extraction DONE ----")

    return {
        **state,
        "extracted_jobs": extracted_jobs,
        "current_task": "validating",
        "status": "validating",
    }
