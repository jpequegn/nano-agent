"""SQLite run logger for nano-agent.

Schema
------
runs
    id          INTEGER PRIMARY KEY AUTOINCREMENT
    task        TEXT      -- the prompt / task description
    status      TEXT      -- 'success' | 'failure'
    failure_reason TEXT   -- free-text reason (used by classifier)
    steps       INTEGER   -- number of agent steps taken
    cost_usd    REAL      -- total cost in USD
    model       TEXT      -- model name used
    started_at  TEXT      -- ISO-8601 timestamp
    finished_at TEXT      -- ISO-8601 timestamp

steps
    id          INTEGER PRIMARY KEY AUTOINCREMENT
    run_id      INTEGER   -- FK → runs.id
    step_num    INTEGER
    tool_name   TEXT      -- tool called (NULL for plain text)
    tool_args   TEXT      -- JSON-encoded args (NULL if no tool)
    tool_result TEXT      -- result/error text
    error       INTEGER   -- 1 if the tool raised an exception
    token_count INTEGER   -- tokens consumed in this step
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

DEFAULT_DB_PATH = Path("agent_runs.db")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task            TEXT    NOT NULL DEFAULT '',
    status          TEXT    NOT NULL DEFAULT 'success',
    failure_reason  TEXT,
    steps           INTEGER NOT NULL DEFAULT 0,
    cost_usd        REAL    NOT NULL DEFAULT 0.0,
    model           TEXT    NOT NULL DEFAULT '',
    started_at      TEXT    NOT NULL,
    finished_at     TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    step_num    INTEGER NOT NULL DEFAULT 0,
    tool_name   TEXT,
    tool_args   TEXT,
    tool_result TEXT,
    error       INTEGER NOT NULL DEFAULT 0,
    token_count INTEGER NOT NULL DEFAULT 0
);
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


@contextmanager
def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection, creating the schema on first use."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def create_run(
    conn: sqlite3.Connection,
    *,
    task: str = "",
    model: str = "",
    started_at: Optional[str] = None,
) -> int:
    """Insert a new run row and return its id."""
    if started_at is None:
        started_at = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO runs (task, status, steps, cost_usd, model, started_at) VALUES (?, 'success', 0, 0.0, ?, ?)",
        (task, model, started_at),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def finish_run(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    status: str = "success",
    failure_reason: Optional[str] = None,
    steps: int = 0,
    cost_usd: float = 0.0,
    finished_at: Optional[str] = None,
) -> None:
    """Update a run row with final stats."""
    if finished_at is None:
        finished_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        UPDATE runs
        SET status = ?, failure_reason = ?, steps = ?, cost_usd = ?, finished_at = ?
        WHERE id = ?
        """,
        (status, failure_reason, steps, cost_usd, finished_at, run_id),
    )
    conn.commit()


def log_step(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    step_num: int = 0,
    tool_name: Optional[str] = None,
    tool_args: Optional[dict[str, Any]] = None,
    tool_result: Optional[str] = None,
    error: bool = False,
    token_count: int = 0,
) -> None:
    """Record a single agent step."""
    args_json = json.dumps(tool_args) if tool_args is not None else None
    conn.execute(
        """
        INSERT INTO steps (run_id, step_num, tool_name, tool_args, tool_result, error, token_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, step_num, tool_name, args_json, tool_result, int(error), token_count),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def list_runs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all runs ordered by id."""
    return conn.execute("SELECT * FROM runs ORDER BY id").fetchall()


def get_steps_for_run(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    """Return all steps for a given run."""
    return conn.execute(
        "SELECT * FROM steps WHERE run_id = ? ORDER BY step_num", (run_id,)
    ).fetchall()
