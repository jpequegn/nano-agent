"""Tests for RunLogger — acceptance criteria for issue #7.

Acceptance criteria:
  - Auto-create DB and schema on first run
  - Log run start/end with status (success/failed/timeout/max_steps)
  - Log each step: type (llm_call / tool_use), inputs, outputs, cost
  - list_recent_runs() returns rows in reverse-chronological order
  - get_run_steps() returns steps in step_num order
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from agent.logger import RunLogger, get_run_steps, list_recent_runs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """A fresh, isolated DB path inside the pytest tmp_path."""
    return tmp_path / "test_runs.db"


@pytest.fixture()
def logger(db_path: Path) -> RunLogger:
    """Return an open RunLogger backed by the isolated test DB."""
    log = RunLogger(task="test task", model="claude-test", db_path=db_path)
    log.open()
    return log


# ---------------------------------------------------------------------------
# DB initialisation
# ---------------------------------------------------------------------------


class TestDatabaseInit:
    def test_db_file_created(self, db_path: Path) -> None:
        """Opening a RunLogger must create the DB file."""
        assert not db_path.exists()
        log = RunLogger(task="t", model="m", db_path=db_path)
        log.open()
        log.close()
        assert db_path.exists()

    def test_db_dir_created(self, tmp_path: Path) -> None:
        """Missing parent directories are created automatically."""
        nested = tmp_path / "a" / "b" / "runs.db"
        log = RunLogger(task="t", model="m", db_path=nested)
        log.open()
        log.close()
        assert nested.exists()

    def test_schema_tables_exist(self, db_path: Path) -> None:
        """Both 'runs' and 'steps' tables must exist after opening."""
        log = RunLogger(task="t", model="m", db_path=db_path)
        log.open()
        log.close()

        conn = sqlite3.connect(str(db_path))
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "runs" in tables
        assert "steps" in tables

    def test_idempotent_schema(self, db_path: Path) -> None:
        """Opening the logger twice must not raise (CREATE TABLE IF NOT EXISTS)."""
        for _ in range(2):
            log = RunLogger(task="t", model="m", db_path=db_path)
            log.open()
            log.close()


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


class TestRunLifecycle:
    def test_run_row_inserted_on_open(self, logger: RunLogger, db_path: Path) -> None:
        logger.close()
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT * FROM runs WHERE id = ?", (logger.run_id,)).fetchall()
        conn.close()
        assert len(rows) == 1

    def test_run_row_has_task_and_model(self, logger: RunLogger, db_path: Path) -> None:
        logger.close()
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT task, model FROM runs WHERE id = ?", (logger.run_id,)
        ).fetchone()
        conn.close()
        assert row[0] == "test task"
        assert row[1] == "claude-test"

    def test_started_at_set(self, logger: RunLogger, db_path: Path) -> None:
        logger.close()
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT started_at FROM runs WHERE id = ?", (logger.run_id,)
        ).fetchone()
        conn.close()
        assert row[0] is not None and len(row[0]) > 0

    def test_finish_success(self, logger: RunLogger, db_path: Path) -> None:
        logger.finish(status="success", total_cost_usd=0.005, final_output="Done!")
        logger.close()

        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT status, total_cost_usd, final_output, finished_at FROM runs WHERE id = ?",
            (logger.run_id,),
        ).fetchone()
        conn.close()

        assert row[0] == "success"
        assert abs(row[1] - 0.005) < 1e-9
        assert row[2] == "Done!"
        assert row[3] is not None

    @pytest.mark.parametrize("status", ["failed", "timeout", "max_steps"])
    def test_finish_various_statuses(
        self, db_path: Path, status: str
    ) -> None:
        log = RunLogger(task="t", model="m", db_path=db_path)
        log.open()
        log.finish(status=status)
        log.close()

        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT status FROM runs WHERE id = ?", (log.run_id,)
        ).fetchone()
        conn.close()
        assert row[0] == status

    def test_context_manager_success(self, db_path: Path) -> None:
        with RunLogger(task="ctx", model="m", db_path=db_path) as log:
            run_id = log.run_id
            log.finish(status="success", final_output="ok")

        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT status FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        conn.close()
        assert row[0] == "success"

    def test_context_manager_exception_marks_failed(self, db_path: Path) -> None:
        run_id: str | None = None
        with pytest.raises(ValueError):
            with RunLogger(task="ctx", model="m", db_path=db_path) as log:
                run_id = log.run_id
                raise ValueError("boom")

        assert run_id is not None
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT status FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        conn.close()
        assert row[0] == "failed"


# ---------------------------------------------------------------------------
# Step logging
# ---------------------------------------------------------------------------


class TestStepLogging:
    def test_log_llm_call(self, logger: RunLogger, db_path: Path) -> None:
        step_id = logger.log_step(
            type="llm_call",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
        )
        logger.close()

        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT * FROM steps WHERE id = ?", (step_id,)).fetchone()
        conn.close()

        assert row is not None
        cols = [desc[0] for desc in conn.description] if False else [
            "id", "run_id", "step_num", "type", "tool_name", "tool_input",
            "tool_output", "input_tokens", "output_tokens", "cost_usd",
        ]
        d = dict(zip(cols, row))
        assert d["type"] == "llm_call"
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50
        assert abs(d["cost_usd"] - 0.001) < 1e-9

    def test_log_tool_use(self, logger: RunLogger, db_path: Path) -> None:
        step_id = logger.log_step(
            type="tool_use",
            tool_name="read_file",
            tool_input={"path": "/etc/hostname"},
            tool_output="myhost",
        )
        logger.close()

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM steps WHERE id = ?", (step_id,)).fetchone()
        conn.close()

        assert dict(row)["tool_name"] == "read_file"
        # tool_input is stored as JSON
        assert json.loads(dict(row)["tool_input"]) == {"path": "/etc/hostname"}
        assert dict(row)["tool_output"] == "myhost"

    def test_step_counter_increments(self, logger: RunLogger, db_path: Path) -> None:
        assert logger.step_count == 0
        logger.log_step(type="llm_call")
        assert logger.step_count == 1
        logger.log_step(type="tool_use", tool_name="x")
        assert logger.step_count == 2
        logger.close()

    def test_step_nums_sequential(self, logger: RunLogger, db_path: Path) -> None:
        for _ in range(3):
            logger.log_step(type="llm_call")
        logger.close()

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT step_num FROM steps WHERE run_id = ? ORDER BY step_num",
            (logger.run_id,),
        ).fetchall()
        conn.close()
        assert [r[0] for r in rows] == [1, 2, 3]

    def test_step_id_returned(self, logger: RunLogger) -> None:
        step_id = logger.log_step(type="llm_call")
        logger.close()
        assert isinstance(step_id, str) and len(step_id) == 36  # UUID format

    def test_log_step_without_open_raises(self, db_path: Path) -> None:
        log = RunLogger(task="t", model="m", db_path=db_path)
        with pytest.raises(RuntimeError, match="not open"):
            log.log_step(type="llm_call")


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


class TestQueryHelpers:
    def test_list_recent_runs_empty_when_no_db(self, tmp_path: Path) -> None:
        missing = tmp_path / "no_such.db"
        assert list_recent_runs(db_path=missing) == []

    def test_list_recent_runs_returns_runs(self, db_path: Path) -> None:
        for i in range(3):
            log = RunLogger(task=f"task {i}", model="m", db_path=db_path)
            log.open()
            log.finish(status="success", total_cost_usd=0.001 * (i + 1))
            log.close()

        runs = list_recent_runs(db_path=db_path)
        assert len(runs) == 3

    def test_list_recent_runs_limit(self, db_path: Path) -> None:
        for i in range(5):
            log = RunLogger(task=f"task {i}", model="m", db_path=db_path)
            log.open()
            log.finish(status="success")
            log.close()

        runs = list_recent_runs(limit=2, db_path=db_path)
        assert len(runs) == 2

    def test_list_recent_runs_ordered_by_started_at_desc(self, db_path: Path) -> None:
        ids = []
        for i in range(3):
            log = RunLogger(task=f"task {i}", model="m", db_path=db_path)
            log.open()
            log.finish(status="success")
            log.close()
            ids.append(log.run_id)

        runs = list_recent_runs(db_path=db_path)
        returned_ids = [r["id"] for r in runs]
        # Most recent (last inserted) should appear first
        assert returned_ids[0] == ids[-1]

    def test_list_recent_runs_includes_step_count(self, db_path: Path) -> None:
        log = RunLogger(task="t", model="m", db_path=db_path)
        log.open()
        log.log_step(type="llm_call")
        log.log_step(type="tool_use", tool_name="x")
        log.finish(status="success")
        log.close()

        runs = list_recent_runs(db_path=db_path)
        assert runs[0]["step_count"] == 2

    def test_get_run_steps_empty_when_no_db(self, tmp_path: Path) -> None:
        missing = tmp_path / "no_such.db"
        assert get_run_steps("any-id", db_path=missing) == []

    def test_get_run_steps_returns_steps_in_order(self, db_path: Path) -> None:
        log = RunLogger(task="t", model="m", db_path=db_path)
        log.open()
        log.log_step(type="llm_call", input_tokens=10, output_tokens=5)
        log.log_step(type="tool_use", tool_name="echo", tool_input={"x": 1})
        log.log_step(type="llm_call", input_tokens=20, output_tokens=10)
        log.finish(status="success")
        log.close()

        steps = get_run_steps(log.run_id, db_path=db_path)
        assert len(steps) == 3
        assert [s["step_num"] for s in steps] == [1, 2, 3]
        assert steps[1]["tool_name"] == "echo"

    def test_get_run_steps_unknown_run(self, db_path: Path) -> None:
        log = RunLogger(task="t", model="m", db_path=db_path)
        log.open()
        log.close()
        steps = get_run_steps("nonexistent-uuid", db_path=db_path)
        assert steps == []


# ---------------------------------------------------------------------------
# Full-replay smoke test  (acceptance criterion)
# ---------------------------------------------------------------------------


class TestFullReplay:
    def test_three_runs_fully_logged(self, db_path: Path) -> None:
        """Run 3 tasks, query the DB, verify full replay of each decision."""
        tasks = [
            ("Write hello.py", [("llm_call", None, None), ("tool_use", "write_file", {"path": "hello.py"})]),
            ("List files", [("llm_call", None, None), ("tool_use", "list_dir", {"path": "."})]),
            ("Add 1+1", [("llm_call", None, None)]),
        ]

        run_ids = []
        for task, steps_spec in tasks:
            log = RunLogger(task=task, model="claude-test", db_path=db_path)
            log.open()
            for step_type, tool_name, tool_input in steps_spec:
                log.log_step(
                    type=step_type,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output="some output",
                    input_tokens=50,
                    output_tokens=20,
                    cost_usd=0.0005,
                )
            log.finish(status="success", total_cost_usd=0.001, final_output="Done")
            log.close()
            run_ids.append(log.run_id)

        # Query and verify
        runs = list_recent_runs(db_path=db_path)
        assert len(runs) == 3

        # Verify each run by its own run_id (index-based, order-independent)
        for i, (run_id, (task, steps_spec)) in enumerate(zip(run_ids, tasks)):
            steps = get_run_steps(run_id, db_path=db_path)
            assert len(steps) == len(steps_spec), (
                f"Run {i} ({task!r}): expected {len(steps_spec)} steps, got {len(steps)}"
            )
            for step, (step_type, tool_name, _) in zip(steps, steps_spec):
                assert step["type"] == step_type
                assert step["tool_name"] == tool_name
