import os
import json
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Shared chat model, built once and reused by every agent.
    Provider selected via LLM_PROVIDER (default 'groq'); currently supports 'groq' and 'openai'.
    To add a provider: add a branch here returning any langchain BaseChatModel."""
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            api_key=os.getenv("GROQ_API_KEY"),
        )
    raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}' (expected: groq, openai)")


def _is_transient(exc: BaseException) -> bool:
    # Groq/OpenAI-style client errors expose a numeric status_code for HTTP failures.
    # Retry on rate limits (429) and server errors (5xx); anything else (bad request,
    # auth) is not going to succeed on retry, so fail fast.
    status = getattr(exc, "status_code", None)
    return status is None or status == 429 or status >= 500


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(_is_transient),
    reraise=True,
)
def invoke_llm(chain, payload: dict):
    """Invoke a langchain Runnable chain with retries on rate limits / transient errors."""
    return chain.invoke(payload)


def parse_llm_json(raw: str) -> Any:
    """Strip optional ```json ... ``` fences and parse the remainder as JSON."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())
