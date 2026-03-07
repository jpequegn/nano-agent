"""CLI entrypoint for nano-agent."""

import argparse
import sys
from pathlib import Path

from agent.logger import DEFAULT_DB_PATH


def cmd_run(args: argparse.Namespace) -> None:
    """Run the agent on a task (not yet implemented)."""
    print("agent run: not yet implemented")
    print(f"  task: {args.task or '(none provided)'}")
    print(f"  model: {args.model}")
    print(f"  max-steps: {args.max_steps}")


def cmd_logs(args: argparse.Namespace) -> None:
    """Show recent run logs (not yet implemented)."""
    print("agent logs: not yet implemented")


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
