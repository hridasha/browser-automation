Manual smoke checks — run individually, by hand, when you need to sanity-check an API key or a
dependency (`python scripts/manual_checks/check_groq.py`, etc). These hit real, billed third-party
APIs (Groq, Tavily) or launch a real browser, so they are **not** part of `pytest` / CI.

For the real automated test suite, see `tests/`.
