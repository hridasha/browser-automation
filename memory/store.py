"""Lightweight run history: persists the outcome of each workflow run to SQLite so
past runs survive process restarts. Not a checkpointer — the live graph state during a
run still only lives in memory; this just records the final (or paused) result of each run.
"""
import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("AGENT_DB_PATH", "agent_history.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    status TEXT NOT NULL,
    human_question TEXT,
    pages_visited INTEGER NOT NULL,
    jobs_found INTEGER NOT NULL,
    jobs_json TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_run(state: dict) -> None:
    """Persist a workflow's final (or paused, e.g. needs_input) state, keyed by run_id."""
    run_id = state.get("run_id")
    if not run_id:
        logger.warning("save_run called without a run_id — skipping persistence")
        return
    try:
        with _connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO runs
                   (run_id, goal, status, human_question, pages_visited, jobs_found, jobs_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    state.get("user_goal", ""),
                    state.get("status", "unknown"),
                    state.get("human_question"),
                    len(state.get("urls_visited", [])),
                    len(state.get("extracted_jobs", [])),
                    json.dumps(state.get("extracted_jobs", [])),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        logger.debug("Saved run %s (status=%s)", run_id, state.get("status"))
    except Exception as e:
        # History is best-effort — never let a storage hiccup fail the actual workflow run.
        logger.error("Failed to save run %s: %s", run_id, e)


def get_run(run_id: str) -> Optional[dict]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["jobs"] = json.loads(result.pop("jobs_json"))
        return result


def list_runs(limit: int = 20, status: Optional[str] = None, offset: int = 0) -> list[dict]:
    where = "WHERE status = ?" if status else ""
    params: tuple = (status,) if status else ()
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""SELECT run_id, goal, status, human_question, pages_visited, jobs_found, created_at
                FROM runs {where} ORDER BY created_at DESC LIMIT ? OFFSET ?""",
            (*params, limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]


def count_runs(status: Optional[str] = None) -> int:
    where = "WHERE status = ?" if status else ""
    params: tuple = (status,) if status else ()
    with _connect() as conn:
        row = conn.execute(f"SELECT COUNT(*) FROM runs {where}", params).fetchone()
        return row[0]
