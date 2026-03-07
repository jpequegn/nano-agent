"""CLI entrypoint for nano-agent."""

import argparse
import sys


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
    """Show failure taxonomy report (not yet implemented)."""
    print("agent report: not yet implemented")


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
    report_parser = subparsers.add_parser("report", help="Show failure taxonomy report")
    report_parser.set_defaults(func=cmd_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
