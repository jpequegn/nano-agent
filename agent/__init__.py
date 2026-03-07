"""nano-agent: a minimal LLM agent with tools, logging, and eval."""

__version__ = "0.1.0"

from agent.tool_registry import ToolNotFoundError, ToolRegistry

__all__ = ["ToolRegistry", "ToolNotFoundError"]
