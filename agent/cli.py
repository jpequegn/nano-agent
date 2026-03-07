"""CLI entrypoint for nano-agent."""

import argparse
import sys


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
