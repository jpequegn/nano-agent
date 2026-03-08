"""Failure taxonomy report for nano-agent.

Reads all runs from SQLite, classifies each failure, and prints a
Markdown table summarising the results.

Failure categories
------------------
infinite_tool_loop   - same tool called 5+ times with identical args
hallucinated_tool     - called a tool that does not exist in the registry
bad_tool_input        - tool raised an exception due to malformed args
context_overflow      - ran out of context window
task_misunderstood    - finished but produced wrong answer (human-labelled)
timeout_max_steps     - hit step or time guard-rails
unknown               - failure with no further detail
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Optional

from agent.logger import DEFAULT_DB_PATH, get_connection, get_steps_for_run, list_runs

# ---------------------------------------------------------------------------
# Known tool names — used to detect hallucinated tools.
# Extend this as real tools are added.
# ---------------------------------------------------------------------------
KNOWN_TOOLS: set[str] = {
    "bash",
    "read_file",
    "write_file",
    "search",
    "python",
}

# Cost threshold above which a run is flagged as expensive
EXPENSIVE_THRESHOLD_USD = 0.10

# How many identical (tool, args) calls in a row to call it an infinite loop
INFINITE_LOOP_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------


def _detect_infinite_loop(steps: list[sqlite3.Row]) -> bool:
    """Return True if the same (tool_name, tool_args) pair appears 5+ times."""
    call_counts: dict[tuple[str, str], int] = defaultdict(int)
    for step in steps:
        if step["tool_name"] is not None:
            key = (step["tool_name"], step["tool_args"] or "")
            call_counts[key] += 1
            if call_counts[key] >= INFINITE_LOOP_THRESHOLD:
                return True
    return False


def _detect_hallucinated_tool(steps: list[sqlite3.Row]) -> bool:
    """Return True if any step called a tool not in KNOWN_TOOLS."""
    for step in steps:
        name = step["tool_name"]
        if name and name not in KNOWN_TOOLS:
            return True
    return False


def _detect_bad_tool_input(steps: list[sqlite3.Row]) -> bool:
    """Return True if any tool raised an exception (error=1)."""
    return any(step["error"] for step in steps)


def classify_run(
    run: sqlite3.Row,
    steps: list[sqlite3.Row],
) -> str:
    """Return the failure category label for a failed run.

    Returns 'success' for runs that completed without error.
    """
    status = run["status"]
    if status == "success":
        return "success"

    reason = (run["failure_reason"] or "").lower()

    # Priority order: most specific first
    if "context" in reason or "context_overflow" in reason or "context length" in reason:
        return "context_overflow"

    if "timeout" in reason or "max_steps" in reason or "max-steps" in reason or "max steps" in reason:
        return "timeout_max_steps"

    if "misunderstood" in reason or "wrong answer" in reason or "incorrect" in reason:
        return "task_misunderstood"

    if _detect_infinite_loop(steps):
        return "infinite_tool_loop"

    if _detect_hallucinated_tool(steps):
        return "hallucinated_tool"

    if _detect_bad_tool_input(steps):
        return "bad_tool_input"

    # Fall back to keyword matching on failure_reason
    if "hallucin" in reason or "does not exist" in reason or "unknown tool" in reason:
        return "hallucinated_tool"

    if "bad input" in reason or "malformed" in reason or "invalid arg" in reason or "exception" in reason:
        return "bad_tool_input"

    if "loop" in reason or "repeat" in reason or "infinite" in reason:
        return "infinite_tool_loop"

    return "unknown"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

_CATEGORY_LABELS: dict[str, str] = {
    "infinite_tool_loop": "Infinite tool loop",
    "hallucinated_tool": "Hallucinated tool",
    "bad_tool_input": "Bad tool input",
    "context_overflow": "Context overflow",
    "task_misunderstood": "Task misunderstood",
    "timeout_max_steps": "Timeout / max-steps",
    "unknown": "Unknown failure",
    "success": "Success",
}

_CATEGORY_ORDER = [
    "infinite_tool_loop",
    "hallucinated_tool",
    "bad_tool_input",
    "context_overflow",
    "task_misunderstood",
    "timeout_max_steps",
    "unknown",
    "success",
]


def _truncate(text: str, max_len: int = 60) -> str:
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def generate_report(db_path: Path | str = DEFAULT_DB_PATH) -> str:
    """Read all runs from *db_path* and return a Markdown report string."""
    lines: list[str] = []

    with get_connection(db_path) as conn:
        runs = list_runs(conn)

        if not runs:
            return "# nano-agent: Failure Taxonomy Report\n\nNo runs found in database.\n"

        # Classify every run
        category_counts: dict[str, int] = defaultdict(int)
        category_examples: dict[str, list[str]] = defaultdict(list)
        expensive_runs: list[tuple[int, str, float]] = []  # (id, task, cost)
        total_runs = len(runs)
        failure_count = 0

        for run in runs:
            run_id = run["id"]
            steps = get_steps_for_run(conn, run_id)
            category = classify_run(run, steps)
            category_counts[category] += 1

            task_snippet = _truncate(run["task"] or f"(run #{run_id})")
            if len(category_examples[category]) < 3:
                category_examples[category].append(task_snippet)

            if run["status"] != "success":
                failure_count += 1

            cost = float(run["cost_usd"] or 0.0)
            if cost > EXPENSIVE_THRESHOLD_USD:
                expensive_runs.append((run_id, task_snippet, cost))

    # ------------------------------------------------------------------
    # Build Markdown output
    # ------------------------------------------------------------------
    lines.append("# nano-agent: Failure Taxonomy Report")
    lines.append("")
    lines.append(f"**Total runs:** {total_runs}  ")
    lines.append(f"**Failures:** {failure_count}  ")
    lines.append(f"**Success rate:** {(total_runs - failure_count) / total_runs * 100:.1f}%")
    lines.append("")

    # Failure breakdown table
    lines.append("## Failure Breakdown")
    lines.append("")
    lines.append("| Failure Type | Count | Example Tasks |")
    lines.append("| --- | --- | --- |")

    has_failures = False
    for cat in _CATEGORY_ORDER:
        if cat == "success":
            continue
        count = category_counts.get(cat, 0)
        if count == 0:
            continue
        has_failures = True
        label = _CATEGORY_LABELS.get(cat, cat)
        examples = "; ".join(f"`{e}`" for e in category_examples[cat])
        lines.append(f"| {label} | {count} | {examples} |")

    if not has_failures:
        lines.append("| — | — | No failures recorded |")

    lines.append("")

    # Success row
    success_count = category_counts.get("success", 0)
    lines.append(f"**Successful runs:** {success_count}/{total_runs}")
    lines.append("")

    # Expensive runs
    lines.append("## Expensive Runs (cost > $0.10)")
    lines.append("")
    if expensive_runs:
        lines.append("| Run ID | Task | Cost (USD) |")
        lines.append("| --- | --- | --- |")
        for run_id, task, cost in sorted(expensive_runs, key=lambda x: -x[2]):
            lines.append(f"| {run_id} | {task} | ${cost:.4f} |")
    else:
        lines.append("_No runs exceeded the $0.10 cost threshold._")
    lines.append("")

    # Category descriptions
    lines.append("## Category Definitions")
    lines.append("")
    lines.append("| Category | Description |")
    lines.append("| --- | --- |")
    lines.append("| Infinite tool loop | Same tool called 5+ times with identical args |")
    lines.append("| Hallucinated tool | Called a tool that does not exist in the registry |")
    lines.append("| Bad tool input | Tool raised an exception due to malformed arguments |")
    lines.append("| Context overflow | Ran out of context window |")
    lines.append("| Task misunderstood | Finished but produced wrong answer (human-labelled) |")
    lines.append("| Timeout / max-steps | Hit step count or time guard-rails |")
    lines.append("")

    return "\n".join(lines)
