from agents.validator import validator_node


def _base_state(**overrides):
    state = {
        "user_goal": "find jobs",
        "extracted_jobs": [],
        "failed_attempts": 0,
        "extraction_fields": ["role", "company", "location", "salary", "apply_url"],
        "max_results": 5,
    }
    state.update(overrides)
    return state


def test_valid_items_pass_through():
    jobs = [{"role": "Engineer", "company": "Acme", "location": "Remote", "salary": None, "apply_url": None}]
    result = validator_node(_base_state(extracted_jobs=jobs))
    assert result["status"] == "done"
    assert result["extracted_jobs"] == jobs


def test_sparse_items_are_dropped_and_retry_triggered():
    jobs = [{"role": "Engineer", "company": None, "location": None, "salary": None, "apply_url": None}]
    result = validator_node(_base_state(extracted_jobs=jobs))
    assert result["extracted_jobs"] == []
    assert result["status"] == "searching"
    assert result["failed_attempts"] == 1
    assert result["urls_visited"] == []
    assert result["pages"] == []


def test_aborts_after_three_failed_attempts():
    result = validator_node(_base_state(extracted_jobs=[], failed_attempts=2))
    assert result["status"] == "error"
    assert result["failed_attempts"] == 3


def test_dedupes_same_role_and_company_case_insensitive():
    jobs = [
        {"role": "Engineer", "company": "Acme", "location": "Remote", "salary": "1", "apply_url": "a"},
        {"role": "engineer", "company": "ACME", "location": "NYC", "salary": "2", "apply_url": "b"},
    ]
    result = validator_node(_base_state(extracted_jobs=jobs))
    assert len(result["extracted_jobs"]) == 1
    assert result["extracted_jobs"][0]["apply_url"] == "a"  # first occurrence kept


def test_low_confidence_item_dropped_despite_enough_fields():
    jobs = [{"role": "Engineer", "company": "Acme", "location": "Remote", "salary": "1", "apply_url": "a", "confidence": 0.1}]
    result = validator_node(_base_state(extracted_jobs=jobs))
    assert result["extracted_jobs"] == []
    assert result["status"] == "searching"


def test_high_confidence_item_with_enough_fields_passes():
    jobs = [{"role": "Engineer", "company": "Acme", "location": "Remote", "salary": "1", "apply_url": "a", "confidence": 0.9}]
    result = validator_node(_base_state(extracted_jobs=jobs))
    assert len(result["extracted_jobs"]) == 1
    assert result["status"] == "done"


def test_missing_confidence_defaults_to_neutral_and_passes():
    jobs = [{"role": "Engineer", "company": "Acme", "location": "Remote", "salary": "1", "apply_url": "a"}]
    result = validator_node(_base_state(extracted_jobs=jobs))
    assert len(result["extracted_jobs"]) == 1
    assert result["status"] == "done"


def test_trims_to_max_results():
    jobs = [
        {"role": f"Engineer {i}", "company": "Acme", "location": "Remote", "salary": "1", "apply_url": "a"}
        for i in range(5)
    ]
    result = validator_node(_base_state(extracted_jobs=jobs, max_results=2))
    assert len(result["extracted_jobs"]) == 2
