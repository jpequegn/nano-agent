#!/usr/bin/env python3
"""Seed a demo SQLite database with 20+ synthetic agent runs.

Usage
-----
    uv run python scripts/seed_demo_db.py [--db PATH]

This creates a realistic-looking agent_runs.db (or the path you specify)
suitable for testing `agent report`.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.logger import DEFAULT_DB_PATH, create_run, finish_run, get_connection, log_step

# ---------------------------------------------------------------------------
# Synthetic run definitions
# ---------------------------------------------------------------------------

_NOW = datetime.now(timezone.utc)


def _ts(offset_minutes: int) -> str:
    return (_NOW - timedelta(minutes=offset_minutes)).isoformat()


# Each entry: (task, status, failure_reason, steps_def, cost_usd)
# steps_def is a list of (tool_name, tool_args_dict, result, error)
RUNS: list[dict] = [
    # ── Successes ──────────────────────────────────────────────────────────
    {
        "task": "Write a Python function that reverses a string",
        "status": "success",
        "failure_reason": None,
        "cost_usd": 0.003,
        "steps": [
            ("bash", {"cmd": "echo hello"}, "hello", False),
            ("write_file", {"path": "reverse.py", "content": "def rev(s): return s[::-1]"}, "ok", False),
        ],
    },
    {
        "task": "Explain the difference between list and tuple in Python",
        "status": "success",
        "failure_reason": None,
        "cost_usd": 0.002,
        "steps": [],
    },
    {
        "task": "Generate a Fibonacci sequence up to 100",
        "status": "success",
        "failure_reason": None,
        "cost_usd": 0.004,
        "steps": [
            ("python", {"code": "print([0,1,1,2,3,5,8,13,21,34,55,89])"}, "[0,1,1,2,3,5,8,13,21,34,55,89]", False),
        ],
    },
    {
        "task": "Sort a list of dictionaries by a key",
        "status": "success",
        "failure_reason": None,
        "cost_usd": 0.002,
        "steps": [],
    },
    {
        "task": "Read a CSV file and compute column averages",
        "status": "success",
        "failure_reason": None,
        "cost_usd": 0.005,
        "steps": [
            ("read_file", {"path": "data.csv"}, "col1,col2\n1,2\n3,4", False),
            ("python", {"code": "import csv; ..."}, "avg col1=2.0 col2=3.0", False),
        ],
    },
    # ── Infinite tool loop ─────────────────────────────────────────────────
    {
        "task": "Find all TODO comments in the codebase",
        "status": "failure",
        "failure_reason": "Infinite tool loop detected",
        "cost_usd": 0.021,
        "steps": [
            ("bash", {"cmd": "grep -r TODO ."}, "TODO: fix bug", False),
            ("bash", {"cmd": "grep -r TODO ."}, "TODO: fix bug", False),
            ("bash", {"cmd": "grep -r TODO ."}, "TODO: fix bug", False),
            ("bash", {"cmd": "grep -r TODO ."}, "TODO: fix bug", False),
            ("bash", {"cmd": "grep -r TODO ."}, "TODO: fix bug", False),
        ],
    },
    {
        "task": "Count lines of code in repository",
        "status": "failure",
        "failure_reason": "loop detected",
        "cost_usd": 0.018,
        "steps": [
            ("bash", {"cmd": "find . -name '*.py' | wc -l"}, "42", False),
            ("bash", {"cmd": "find . -name '*.py' | wc -l"}, "42", False),
            ("bash", {"cmd": "find . -name '*.py' | wc -l"}, "42", False),
            ("bash", {"cmd": "find . -name '*.py' | wc -l"}, "42", False),
            ("bash", {"cmd": "find . -name '*.py' | wc -l"}, "42", False),
            ("bash", {"cmd": "find . -name '*.py' | wc -l"}, "42", False),
        ],
    },
    # ── Hallucinated tool ──────────────────────────────────────────────────
    {
        "task": "Search the web for Python best practices",
        "status": "failure",
        "failure_reason": "Called unknown tool: web_search",
        "cost_usd": 0.006,
        "steps": [
            ("web_search", {"query": "Python best practices 2024"}, "ToolNotFoundError: web_search", True),
        ],
    },
    {
        "task": "Get current weather in Paris",
        "status": "failure",
        "failure_reason": "hallucinated tool call",
        "cost_usd": 0.004,
        "steps": [
            ("get_weather", {"city": "Paris"}, "ToolNotFoundError: get_weather", True),
        ],
    },
    {
        "task": "Send an email summary of today's tasks",
        "status": "failure",
        "failure_reason": "unknown tool",
        "cost_usd": 0.005,
        "steps": [
            ("send_email", {"to": "user@example.com", "body": "..."}, "ToolNotFoundError: send_email", True),
        ],
    },
    # ── Bad tool input ─────────────────────────────────────────────────────
    {
        "task": "Append content to a log file",
        "status": "failure",
        "failure_reason": "bad input to write_file",
        "cost_usd": 0.007,
        "steps": [
            ("write_file", {"path": None, "content": "log entry"}, "TypeError: path must be a string", True),
        ],
    },
    {
        "task": "Run a shell command with a timeout",
        "status": "failure",
        "failure_reason": "malformed args",
        "cost_usd": 0.008,
        "steps": [
            ("bash", {"cmd": ["ls", "-la"]}, "TypeError: cmd must be str, got list", True),
        ],
    },
    # ── Context overflow ───────────────────────────────────────────────────
    {
        "task": "Summarise a 500-page PDF document",
        "status": "failure",
        "failure_reason": "context overflow: exceeded max context length",
        "cost_usd": 0.145,
        "steps": [
            ("read_file", {"path": "book.pdf"}, "<500 pages of text...>", False),
        ],
    },
    {
        "task": "Refactor an entire 10 000-line legacy codebase",
        "status": "failure",
        "failure_reason": "context length exceeded",
        "cost_usd": 0.213,
        "steps": [
            ("read_file", {"path": "legacy.py"}, "<10000 lines>", False),
        ],
    },
    # ── Task misunderstood ─────────────────────────────────────────────────
    {
        "task": "What is the capital of Australia?",
        "status": "failure",
        "failure_reason": "wrong answer: agent said Sydney instead of Canberra",
        "cost_usd": 0.002,
        "steps": [],
    },
    {
        "task": "Compute 17 * 34",
        "status": "failure",
        "failure_reason": "incorrect result returned",
        "cost_usd": 0.003,
        "steps": [],
    },
    # ── Timeout / max-steps ────────────────────────────────────────────────
    {
        "task": "Implement a full REST API with auth, tests, and deployment config",
        "status": "failure",
        "failure_reason": "max_steps reached (20/20)",
        "cost_usd": 0.089,
        "steps": [("bash", {"cmd": f"step {i}"}, "ok", False) for i in range(20)],
    },
    {
        "task": "Build a complete e-commerce checkout flow",
        "status": "failure",
        "failure_reason": "timeout: agent exceeded wall-clock limit",
        "cost_usd": 0.052,
        "steps": [("bash", {"cmd": f"step {i}"}, "ok", False) for i in range(15)],
    },
    # ── More successes to hit 20+ total ───────────────────────────────────
    {
        "task": "Convert a list of strings to uppercase",
        "status": "success",
        "failure_reason": None,
        "cost_usd": 0.001,
        "steps": [],
    },
    {
        "task": "Write unit tests for a stack data structure",
        "status": "success",
        "failure_reason": None,
        "cost_usd": 0.006,
        "steps": [
            ("write_file", {"path": "test_stack.py", "content": "..."}, "ok", False),
            ("bash", {"cmd": "python -m pytest test_stack.py"}, "3 passed", False),
        ],
    },
    {
        "task": "Explain async/await in Python with examples",
        "status": "success",
        "failure_reason": None,
        "cost_usd": 0.004,
        "steps": [],
    },
    {
        "task": "Fix indentation errors in script.py",
        "status": "success",
        "failure_reason": None,
        "cost_usd": 0.003,
        "steps": [
            ("read_file", {"path": "script.py"}, "def foo():\n pass", False),
            ("write_file", {"path": "script.py", "content": "def foo():\n    pass"}, "ok", False),
        ],
    },
]


# ---------------------------------------------------------------------------
# Seeding logic
# ---------------------------------------------------------------------------


def seed(db_path: Path) -> None:
    print(f"Seeding {db_path} with {len(RUNS)} synthetic runs …")
    with get_connection(db_path) as conn:
        for idx, spec in enumerate(RUNS):
            started = _ts((len(RUNS) - idx) * 5)  # spaced 5 min apart
            run_id = create_run(
                conn,
                task=spec["task"],
                model="claude-3-5-haiku-20241022",
                started_at=started,
            )
            # Log steps
            for step_num, (tool_name, tool_args, tool_result, error) in enumerate(spec["steps"]):
                log_step(
                    conn,
                    run_id,
                    step_num=step_num,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=str(tool_result),
                    error=error,
                    token_count=200 + step_num * 50,
                )
            finish_run(
                conn,
                run_id,
                status=spec["status"],
                failure_reason=spec["failure_reason"],
                steps=len(spec["steps"]),
                cost_usd=spec["cost_usd"],
            )
    print(f"Done. {len(RUNS)} runs written to {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed a demo agent runs database")
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        metavar="PATH",
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()
    seed(Path(args.db))


if __name__ == "__main__":
    main()
