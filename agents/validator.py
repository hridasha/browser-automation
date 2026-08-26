import logging

from memory.state_manager import AgentState

logger = logging.getLogger(__name__)

# Below this, an item is dropped even if enough fields are filled — the model itself
# flagged the extraction as unreliable (ambiguous/boilerplate page content).
MIN_CONFIDENCE = 0.3


def _dedupe(items: list[dict]) -> list[dict]:
    """Drop items that share the same (role, company), case-insensitive — the same job
    often gets extracted more than once when several search results point at it."""
    seen = set()
    deduped = []
    for item in items:
        key = (str(item.get("role", "")).strip().lower(), str(item.get("company", "")).strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def validator_node(state: AgentState) -> AgentState:
    extracted_jobs = state.get("extracted_jobs", [])
    failed_attempts = state.get("failed_attempts", 0)
    extraction_fields = state.get("extraction_fields", [])
    max_results = state.get("max_results", 5)
    valid_items = []
    invalid_count = 0

    # require at least 2 non-null fields from whatever the planner asked for
    min_filled = max(2, len(extraction_fields) // 2)

    logger.info("---- Validator START ----")
    logger.info("Items to check      : %d", len(extracted_jobs))
    logger.debug("Tracked fields      : %s", extraction_fields)
    logger.debug("Min non-null needed : %d", min_filled)

    for i, item in enumerate(extracted_jobs, 1):
        filled = [f for f in extraction_fields if item.get(f) not in (None, "", "null", "N/A")]
        confidence = item.get("confidence", 0.5)
        enough_fields = len(filled) >= min_filled
        confident_enough = confidence >= MIN_CONFIDENCE

        if enough_fields and confident_enough:
            valid_items.append(item)
        else:
            empty = [f for f in extraction_fields if f not in filled]
            reason = [] if enough_fields else [f"only {len(filled)}/{min_filled} fields filled, empty: {empty}"]
            if not confident_enough:
                reason.append(f"confidence {confidence:.2f} < {MIN_CONFIDENCE}")
            logger.debug("[%d] INVALID — %s", i, "; ".join(reason))
            invalid_count += 1

    logger.info("Valid   : %d", len(valid_items))
    logger.info("Invalid : %d", invalid_count)

    if len(valid_items) == 0:
        failed_attempts += 1
        new_status = "error" if failed_attempts >= 3 else "searching"
        logger.warning("No valid items — failed_attempts now %d, status → '%s'", failed_attempts, new_status)
        logger.info("---- Validator DONE (retry/abort) ----")
        return {
            **state,
            "extracted_jobs": [],
            "failed_attempts": failed_attempts,
            "status": new_status,
            "current_task": "retrying",
            "urls_visited": [],
            "pages": [],
        }

    deduped = _dedupe(valid_items)
    if len(deduped) != len(valid_items):
        logger.info("Deduped %d → %d (removed same role+company matches)", len(valid_items), len(deduped))

    final_items = deduped[:max_results]
    if len(final_items) != len(deduped):
        logger.info("Trimmed %d → %d to respect max_results=%d", len(deduped), len(final_items), max_results)

    logger.info("Setting status → 'done'")
    logger.info("---- Validator DONE (success) ----")
    return {
        **state,
        "extracted_jobs": final_items,
        "failed_attempts": failed_attempts,
        "status": "done",
        "current_task": "done",
    }
