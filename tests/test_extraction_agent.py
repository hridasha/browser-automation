import json
from unittest.mock import patch, MagicMock

from agents.extraction_agent import extraction_node


def _fake_response(payload):
    resp = MagicMock()
    resp.content = json.dumps(payload)
    return resp


def _base_state(**overrides):
    state = {
        "pages": [{"url": "https://example.com/job", "content": "some job posting text"}],
        "extraction_fields": ["role", "company", "location", "salary", "apply_url"],
        "extracted_jobs": [],
    }
    state.update(overrides)
    return state


def test_normalizes_field_name_variants_from_model_drift():
    # A model that ignores the "use these exact field names" instruction and returns
    # its own naming (seen in practice: position_title / link instead of role / apply_url).
    raw_item = {"company": "Acme", "position_title": "ML Intern", "location": "Remote", "link": "https://acme.example/apply"}
    with patch("agents.extraction_agent.invoke_llm", return_value=_fake_response(raw_item)):
        result = extraction_node(_base_state())

    job = result["extracted_jobs"][0]
    assert job["role"] == "ML Intern"
    assert job["apply_url"] == "https://acme.example/apply"
    assert "position_title" not in job
    assert "link" not in job


def test_normalizes_title_case_field_names_from_planner_chosen_labels():
    # The planner is free to phrase extraction_fields however it wants (e.g. "Job Title",
    # "Company", "Application Link" instead of role/company/apply_url), and the model
    # echoes those exact labels back as JSON keys. This must still resolve to canonical
    # names, or the validator sees every field as missing.
    raw_item = {"Job Title": "Data Analyst Intern", "Company": "Acme", "Location": "Remote", "Application Link": "https://acme.example/apply"}
    with patch("agents.extraction_agent.invoke_llm", return_value=_fake_response(raw_item)):
        result = extraction_node(_base_state())

    job = result["extracted_jobs"][0]
    assert job["role"] == "Data Analyst Intern"
    assert job["company"] == "Acme"
    assert job["location"] == "Remote"
    assert job["apply_url"] == "https://acme.example/apply"


def test_prefers_filled_value_when_canonical_key_is_null_but_alias_has_data():
    raw_item = {"company": None, "company_name": "Acme", "role": "Engineer"}
    with patch("agents.extraction_agent.invoke_llm", return_value=_fake_response(raw_item)):
        result = extraction_node(_base_state())

    assert result["extracted_jobs"][0]["company"] == "Acme"


def test_leaves_canonical_field_names_untouched():
    raw_item = {"company": "Acme", "role": "Engineer", "apply_url": "https://acme.example"}
    with patch("agents.extraction_agent.invoke_llm", return_value=_fake_response(raw_item)):
        result = extraction_node(_base_state())

    job = result["extracted_jobs"][0]
    assert job["role"] == "Engineer"
    assert job["apply_url"] == "https://acme.example"


def test_confidence_passed_through_and_clamped():
    raw_item = {"company": "Acme", "role": "Engineer", "confidence": 1.7}
    with patch("agents.extraction_agent.invoke_llm", return_value=_fake_response(raw_item)):
        result = extraction_node(_base_state())

    assert result["extracted_jobs"][0]["confidence"] == 1.0


def test_confidence_defaults_when_missing_or_invalid():
    raw_item = {"company": "Acme", "role": "Engineer", "confidence": "unsure"}
    with patch("agents.extraction_agent.invoke_llm", return_value=_fake_response(raw_item)):
        result = extraction_node(_base_state())

    assert result["extracted_jobs"][0]["confidence"] == 0.5
