#!/usr/bin/env python3
"""
eval/run_eval.py — Run all 20 evaluation tasks and save results.

Since the agent loop (issue #6) is not yet implemented, this script uses
a minimal inline harness that:
  1. Loads tasks from eval/tasks.json
  2. For each task, builds a simulated agent with the tool registry
  3. Calls the Anthropic API (claude-3-5-haiku) with tool use enabled
  4. Executes tool calls in a loop until the model returns a final answer
  5. Saves per-task results to eval/results/<task_id>.json
  6. Writes a summary CSV to eval/results/summary.csv

Usage:
    uv run python eval/run_eval.py
    uv run python eval/run_eval.py --task FM-01        # single task
    uv run python eval/run_eval.py --dry-run           # list tasks, no API calls
    uv run python eval/run_eval.py --max-steps 10      # override step limit
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
EVAL_DIR = Path(__file__).parent
TASKS_FILE = EVAL_DIR / "tasks.json"
RESULTS_DIR = EVAL_DIR / "results"
FIXTURES_DIR = EVAL_DIR / "fixtures"
OUTPUT_DIR = FIXTURES_DIR / "output"

# ---------------------------------------------------------------------------
# Minimal tool implementations for the eval harness
# These mirror what a full agent would expose, scoped to the eval context.
# ---------------------------------------------------------------------------


def tool_read_file(path: str) -> str:
    """Read a file and return its contents as a string."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return p.read_text()


def tool_write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Written {len(content)} bytes to {path}"


def tool_append_file(path: str, content: str) -> str:
    """Append content to a file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(content)
    return f"Appended {len(content)} bytes to {path}"


def tool_run_bash(command: str) -> str:
    """Run a bash command and return stdout + stderr. Timeout: 15s."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(REPO_ROOT),
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        output += f"\n[exit_code]: {result.returncode}"
        return output.strip()
    except subprocess.TimeoutExpired:
        return "[error]: Command timed out after 15 seconds"


def tool_list_files(directory: str) -> str:
    """List files in a directory (non-recursive)."""
    p = Path(directory)
    if not p.exists():
        return f"[error]: Directory not found: {directory}"
    items = list(p.iterdir())
    lines = [str(item.relative_to(REPO_ROOT)) for item in sorted(items)]
    return "\n".join(lines) if lines else "(empty directory)"


def tool_path_exists(path: str) -> str:
    """Check whether a file or directory exists."""
    p = Path(path)
    if p.exists():
        kind = "directory" if p.is_dir() else "file"
        return f"EXISTS: {path} is a {kind}"
    return f"NOT_EXISTS: {path} does not exist"


# ---------------------------------------------------------------------------
# Tool registry for the harness
# ---------------------------------------------------------------------------

TOOLS: dict[str, Any] = {
    "read_file": {
        "func": tool_read_file,
        "description": "Read a file and return its full contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"}
            },
            "required": ["path"],
        },
    },
    "write_file": {
        "func": tool_write_file,
        "description": "Write content to a file (creates parent dirs if needed).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Destination file path"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
    "append_file": {
        "func": tool_append_file,
        "description": "Append content to an existing file (or create it).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to append to"},
                "content": {"type": "string", "description": "Content to append"},
            },
            "required": ["path", "content"],
        },
    },
    "run_bash": {
        "func": tool_run_bash,
        "description": "Run a bash command and return stdout, stderr, and exit code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run"}
            },
            "required": ["command"],
        },
    },
    "list_files": {
        "func": tool_list_files,
        "description": "List files and directories in a given directory path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory path to list",
                }
            },
            "required": ["directory"],
        },
    },
    "path_exists": {
        "func": tool_path_exists,
        "description": "Check whether a file or directory path exists.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to check"}
            },
            "required": ["path"],
        },
    },
}


def get_api_tools() -> list[dict[str, Any]]:
    """Return the tools list in Anthropic API format."""
    return [
        {
            "name": name,
            "description": entry["description"],
            "input_schema": entry["input_schema"],
        }
        for name, entry in TOOLS.items()
    ]


def execute_tool(name: str, tool_input: dict[str, Any]) -> str:
    """Execute a tool by name. Returns a string result."""
    if name not in TOOLS:
        return f"[error]: Unknown tool '{name}'"
    try:
        result = TOOLS[name]["func"](**tool_input)
        return str(result)
    except Exception as e:
        return f"[error]: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Agent harness
# ---------------------------------------------------------------------------


def run_task(
    task: dict[str, Any],
    client: Any,
    model: str = "claude-3-5-haiku-20241022",
    max_steps: int = 15,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run a single task through the agent loop.

    Returns a result dict with:
      - task_id, title, category
      - status: "ok" | "max_steps" | "error"
      - steps: number of tool calls made
      - final_response: the model's last text output
      - tool_calls: list of {tool, input, output} dicts
      - duration_s: wall-clock seconds
      - error: error message if status == "error"
    """
    task_id = task["id"]
    prompt = task["prompt"]

    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    tool_calls_log: list[dict[str, Any]] = []
    steps = 0
    final_response = ""
    status = "ok"
    error_msg = ""
    start = time.time()

    if verbose:
        print(f"\n  Prompt: {prompt[:100]}...")

    try:
        while steps < max_steps:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                tools=get_api_tools(),
                messages=messages,
            )

            # Accumulate text content
            text_parts = []
            tool_use_blocks = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_use_blocks.append(block)

            if text_parts:
                final_response = "\n".join(text_parts)

            # If no tool calls, we're done
            if not tool_use_blocks or response.stop_reason == "end_turn":
                break

            # Append assistant turn
            messages.append({"role": "assistant", "content": response.content})

            # Execute tool calls and build tool_results
            tool_results = []
            for tool_block in tool_use_blocks:
                steps += 1
                tool_name = tool_block.name
                tool_input = tool_block.input
                tool_output = execute_tool(tool_name, tool_input)

                tool_calls_log.append(
                    {
                        "step": steps,
                        "tool": tool_name,
                        "input": tool_input,
                        "output": tool_output[:500],  # truncate long outputs
                    }
                )

                if verbose:
                    print(f"    [step {steps}] {tool_name}({tool_input}) → {tool_output[:80]}...")

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": tool_output,
                    }
                )

                if steps >= max_steps:
                    status = "max_steps"
                    break

            # Append tool results turn
            messages.append({"role": "user", "content": tool_results})

            if status == "max_steps":
                break

        # If we exited the loop with tool calls still pending (max_steps hit),
        # get a final response
        if status == "max_steps":
            final_response = f"[max_steps reached after {steps} tool calls]"

    except Exception as e:
        status = "error"
        error_msg = f"{type(e).__name__}: {e}"
        if verbose:
            print(f"    [ERROR] {error_msg}")

    duration = time.time() - start

    return {
        "task_id": task_id,
        "title": task["title"],
        "category": task["category"],
        "prompt": prompt,
        "expected_output": task["expected_output"],
        "status": status,
        "steps": steps,
        "final_response": final_response,
        "tool_calls": tool_calls_log,
        "duration_s": round(duration, 2),
        "error": error_msg,
        "grading": task.get("grading", {}),
        "grade": "ungraded",  # set by human grader
        "notes": "",
    }


# ---------------------------------------------------------------------------
# Dry-run mode (no API calls)
# ---------------------------------------------------------------------------


def dry_run(tasks: list[dict[str, Any]]) -> None:
    """Print task list without making any API calls."""
    print(f"\n{'─' * 70}")
    print(f"  DRY RUN — {len(tasks)} tasks loaded from {TASKS_FILE}")
    print(f"{'─' * 70}")

    categories: dict[str, list[str]] = {}
    for task in tasks:
        cat = task["category"]
        categories.setdefault(cat, []).append(task["id"])

    for cat, ids in categories.items():
        print(f"\n  {cat} ({len(ids)} tasks):")
        for tid in ids:
            task = next(t for t in tasks if t["id"] == tid)
            print(f"    {tid}: {task['title']}")

    print(f"\n  Tools available: {', '.join(TOOLS.keys())}")
    print(f"  Results will be saved to: {RESULTS_DIR}/\n")


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------


def write_summary(results: list[dict[str, Any]]) -> Path:
    """Write a summary CSV of all results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "summary.csv"

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "category",
                "title",
                "status",
                "steps",
                "duration_s",
                "grade",
                "error",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "task_id": r["task_id"],
                    "category": r["category"],
                    "title": r["title"],
                    "status": r["status"],
                    "steps": r["steps"],
                    "duration_s": r["duration_s"],
                    "grade": r["grade"],
                    "error": r["error"][:100] if r["error"] else "",
                }
            )

    return summary_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run nano-agent evaluation tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task",
        help="Run only the task with this ID (e.g. FM-01)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List tasks without making API calls",
    )
    parser.add_argument(
        "--model",
        default="claude-3-5-haiku-20241022",
        help="Anthropic model to use",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        dest="max_steps",
        help="Max tool-call steps per task (default: 15)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print step-by-step progress",
    )
    args = parser.parse_args()

    # Load tasks
    with TASKS_FILE.open() as f:
        data = json.load(f)
    tasks = data["tasks"]

    # Filter if --task provided
    if args.task:
        tasks = [t for t in tasks if t["id"] == args.task]
        if not tasks:
            print(f"[error] No task with ID '{args.task}' found.")
            sys.exit(1)

    # Dry-run: just list tasks
    if args.dry_run:
        dry_run(tasks)
        return

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[error] ANTHROPIC_API_KEY not set. Export it or add to .env")
        sys.exit(1)

    # Import Anthropic client
    try:
        import anthropic  # type: ignore[import]
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        print("[error] anthropic package not installed. Run: uv sync")
        sys.exit(1)

    # Ensure output directories exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run tasks
    print(f"\n{'═' * 70}")
    print(f"  nano-agent eval  •  {len(tasks)} tasks  •  model: {args.model}")
    print(f"  started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 70}\n")

    results: list[dict[str, Any]] = []
    for i, task in enumerate(tasks, 1):
        task_id = task["id"]
        print(f"  [{i:02d}/{len(tasks):02d}] {task_id}: {task['title']}", end="", flush=True)

        result = run_task(
            task=task,
            client=client,
            model=args.model,
            max_steps=args.max_steps,
            verbose=args.verbose,
        )
        results.append(result)

        # Status indicator
        status_icon = {"ok": "✓", "max_steps": "⚠", "error": "✗"}.get(result["status"], "?")
        print(f"  {status_icon}  ({result['steps']} steps, {result['duration_s']}s)")

        # Save individual result
        result_path = RESULTS_DIR / f"{task_id}.json"
        with result_path.open("w") as f:
            json.dump(result, f, indent=2)

    # Write summary
    summary_path = write_summary(results)

    # Print summary table
    print(f"\n{'─' * 70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'─' * 70}")
    ok = sum(1 for r in results if r["status"] == "ok")
    max_steps = sum(1 for r in results if r["status"] == "max_steps")
    errors = sum(1 for r in results if r["status"] == "error")
    total_steps = sum(r["steps"] for r in results)
    total_time = sum(r["duration_s"] for r in results)

    print(f"  Total tasks:   {len(results)}")
    print(f"  Completed OK:  {ok}")
    print(f"  Max steps hit: {max_steps}")
    print(f"  Errors:        {errors}")
    print(f"  Total steps:   {total_steps}")
    print(f"  Total time:    {total_time:.1f}s")
    print(f"\n  Results saved to: {RESULTS_DIR}/")
    print(f"  Summary CSV:      {summary_path}")
    print(f"\n  Next: human-grade results in {summary_path}")
    print(f"        (set 'grade' column to pass/partial/fail and add notes)")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
