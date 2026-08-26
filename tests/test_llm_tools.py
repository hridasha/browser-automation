import pytest
from tools.llm_tools import parse_llm_json


def test_parses_plain_json():
    assert parse_llm_json('{"a": 1}') == {"a": 1}


def test_strips_fenced_json_block():
    raw = '```json\n{"a": 1}\n```'
    assert parse_llm_json(raw) == {"a": 1}


def test_strips_fenced_block_without_json_tag():
    raw = '```\n[1, 2, 3]\n```'
    assert parse_llm_json(raw) == [1, 2, 3]


def test_invalid_json_raises():
    with pytest.raises(Exception):
        parse_llm_json("not json at all")
