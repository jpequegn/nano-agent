"""CLI entrypoint for nano-agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent.logger import DEFAULT_DB_PATH

from agent.logger import list_recent_runs


def cmd_run(args: argparse.Namespace) -> None:
    """Run the agent on a task."""
    from agent.agent import Agent
    from agent.tool_registry import ToolRegistry

    task = args.task
    if not task:
        print("Error: a task description is required.", file=sys.stderr)
        sys.exit(1)

    # Build a minimal registry with a couple of built-in tools for the CLI
    registry = ToolRegistry()

    @registry.tool(description="Evaluate a Python expression and return the result as a string.")
    def python_eval(expression: str) -> str:
        return str(eval(expression))  # noqa: S307  # intentional for demo

    agent = Agent(registry=registry, model=args.model, max_steps=args.max_steps)

    print(f"Running agent (model={args.model}, max_steps={args.max_steps})…")
    result = agent.run(task)
    print(result)

    # Print cost summary after each run
    if hasattr(agent, "last_run_cost"):
        print()
        print(agent.last_run_cost.summary())


def cmd_logs(args: argparse.Namespace) -> None:
    """List recent runs from the SQLite database."""
    runs = list_recent_runs(limit=args.limit)

    if not runs:
        print("No runs recorded yet. Run `agent run <task>` to get started.")
        return

    # Header
    header = (
        f"{'ID':36}  {'STATUS':10}  {'STEPS':5}  {'COST (USD)':10}  "
        f"{'MODEL':30}  {'STARTED':26}  TASK"
    )
    print(header)
    print("-" * len(header))

    for run in runs:
        run_id = run["id"]
        status = run["status"] or "running"
        steps = run["step_count"]
        cost = run["total_cost_usd"]
        cost_str = f"${cost:.6f}" if cost is not None else "n/a"
        model = (run["model"] or "")[:30]
        started = (run["started_at"] or "")[:26]
        task = (run["task"] or "")[:60]

        print(
            f"{run_id}  {status:<10}  {steps:5}  {cost_str:>10}  "
            f"{model:<30}  {started:<26}  {task}"
        )


def cmd_report(args: argparse.Namespace) -> None:
    """Read all runs from SQLite and print a failure taxonomy report."""
    from agent.report import generate_report

    db_path = Path(args.db)
    if not db_path.exists():
        print(
            f"[agent report] Database not found: {db_path}\n"
            "Run some tasks first (agent run ...) or seed the DB with scripts/seed_demo_db.py",
            file=sys.stderr,
        )
        sys.exit(1)

    report = generate_report(db_path)
    print(report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="nano-agent: a minimal LLM coding agent",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    # agent run
    run_parser = subparsers.add_parser("run", help="Run the agent on a task")
    run_parser.add_argument("task", nargs="?", help="Task description to run")
    run_parser.add_argument(
        "--model",
        default="claude-3-5-haiku-20241022",
        help="Model to use (default: claude-3-5-haiku-20241022)",
    )
    run_parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        dest="max_steps",
        help="Maximum number of agent steps (default: 20)",
    )
    run_parser.set_defaults(func=cmd_run)

    # agent logs
    logs_parser = subparsers.add_parser("logs", help="List recent runs")
    logs_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of runs to display (default: 20)",
    )
    logs_parser.set_defaults(func=cmd_logs)

    # agent report
    report_parser = subparsers.add_parser(
        "report",
        help="Show failure taxonomy report from SQLite run history",
    )
    report_parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        metavar="PATH",
        help=f"Path to the SQLite database (default: {DEFAULT_DB_PATH})",
    )
    report_parser.set_defaults(func=cmd_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
