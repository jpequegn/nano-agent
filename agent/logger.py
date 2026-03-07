"""SQLite run logger: records every decision and tool result for replay/analysis.

Database location: ~/.nano-agent/runs.db

Schema
------
runs  : one row per agent run (task, model, status, cost, …)
steps : one row per LLM call or tool-use event within a run
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_DIR = Path.home() / ".nano-agent"
DB_PATH = DB_DIR / "runs.db"

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    task            TEXT,
    model           TEXT,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    status          TEXT,
    total_cost_usd  REAL,
    final_output    TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    step_num        INTEGER NOT NULL,
    type            TEXT NOT NULL,   -- 'llm_call' | 'tool_use'
    tool_name       TEXT,
    tool_input      TEXT,            -- JSON
    tool_output     TEXT,            -- JSON or plain text
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cost_usd        REAL
);
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _json_or_str(value: Any) -> str | None:
    """Serialise *value* to a JSON string, or return None if value is None."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# RunLogger
# ---------------------------------------------------------------------------


class RunLogger:
    """Logs a single agent run (and all its steps) to SQLite.

    Typical usage::

        with RunLogger(task="Write hello.py", model="claude-3-5-haiku-20241022") as log:
            step_id = log.log_step(type="llm_call", input_tokens=100, output_tokens=50, cost_usd=0.001)
            log.log_step(type="tool_use", tool_name="read_file", tool_input={"path": "x"}, tool_output="…")
            log.finish(status="success", total_cost_usd=0.001, final_output="Done")
    """

    def __init__(
        self,
        task: str,
        model: str,
        *,
        db_path: Path = DB_PATH,
    ) -> None:
        self.task = task
        self.model = model
        self.db_path = db_path
        self.run_id = str(uuid.uuid4())
        self._step_counter = 0
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> RunLogger:
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # If caller didn't call finish() explicitly (e.g. exception), mark failed
        if exc_type is not None:
            try:
                self.finish(status="failed")
            except Exception:
                pass
        self.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Ensure the DB directory and schema exist, then start the run row."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.execute(
            "INSERT INTO runs (id, task, model, started_at) VALUES (?, ?, ?, ?)",
            (self.run_id, self.task, self.model, _now_iso()),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying DB connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def finish(
        self,
        *,
        status: str,
        total_cost_usd: float | None = None,
        final_output: str | None = None,
    ) -> None:
        """Update the run row with end-of-run metadata.

        Args:
            status: One of ``"success"``, ``"failed"``, ``"timeout"``, ``"max_steps"``.
            total_cost_usd: Cumulative USD cost across all steps.
            final_output: The agent's final answer / output text.
        """
        if self._conn is None:
            raise RuntimeError("RunLogger is not open — call open() first.")
        self._conn.execute(
            """
            UPDATE runs
            SET finished_at = ?, status = ?, total_cost_usd = ?, final_output = ?
            WHERE id = ?
            """,
            (_now_iso(), status, total_cost_usd, final_output, self.run_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Step logging
    # ------------------------------------------------------------------

    def log_step(
        self,
        *,
        type: str,  # noqa: A002  (shadow builtin intentionally for readability)
        tool_name: str | None = None,
        tool_input: Any = None,
        tool_output: Any = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> str:
        """Insert one step row and return its generated ID.

        Args:
            type: ``"llm_call"`` or ``"tool_use"``.
            tool_name: Name of the tool (only for ``tool_use`` steps).
            tool_input: Tool input dict / value (serialised to JSON).
            tool_output: Tool output / LLM response (serialised to JSON or kept as str).
            input_tokens: Prompt token count (for ``llm_call`` steps).
            output_tokens: Completion token count (for ``llm_call`` steps).
            cost_usd: Cost of this individual step in USD.

        Returns:
            The UUID string assigned to this step row.
        """
        if self._conn is None:
            raise RuntimeError("RunLogger is not open — call open() first.")

        self._step_counter += 1
        step_id = str(uuid.uuid4())

        self._conn.execute(
            """
            INSERT INTO steps
                (id, run_id, step_num, type, tool_name, tool_input, tool_output,
                 input_tokens, output_tokens, cost_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                step_id,
                self.run_id,
                self._step_counter,
                type,
                tool_name,
                _json_or_str(tool_input),
                _json_or_str(tool_output),
                input_tokens,
                output_tokens,
                cost_usd,
            ),
        )
        self._conn.commit()
        return step_id

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        """Number of steps logged so far in this run."""
        return self._step_counter


# ---------------------------------------------------------------------------
# Query helpers (used by `agent logs`)
# ---------------------------------------------------------------------------


def list_recent_runs(
    limit: int = 20,
    *,
    db_path: Path = DB_PATH,
) -> list[dict[str, Any]]:
    """Return the *limit* most-recent runs as a list of plain dicts.

    Returns an empty list if the database file does not exist yet.
    """
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        # Ensure schema exists (idempotent)
        conn.executescript(_DDL)
        rows = conn.execute(
            """
            SELECT r.id, r.task, r.model, r.started_at, r.finished_at,
                   r.status, r.total_cost_usd, r.final_output,
                   COUNT(s.id) AS step_count
            FROM runs r
            LEFT JOIN steps s ON s.run_id = r.id
            GROUP BY r.id
            ORDER BY r.started_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_run_steps(run_id: str, *, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    """Return all steps for a given run, ordered by step_num."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_DDL)
        rows = conn.execute(
            """
            SELECT * FROM steps WHERE run_id = ? ORDER BY step_num
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
