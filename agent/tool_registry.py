"""Tool registry: register, describe, and execute tools for the agent."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ToolNotFoundError(KeyError):
    """Raised when an unknown tool name is looked up in the registry."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Tool not found: {name!r}")


# ---------------------------------------------------------------------------
# Type → JSON Schema helpers
# ---------------------------------------------------------------------------

_PYTHON_TYPE_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    bytes: "string",
}


def _python_type_to_json_type(annotation: Any) -> str:
    """Convert a Python type annotation to a JSON Schema type string.

    Handles plain types, ``Optional[X]`` (``Union[X, None]``), and falls
    back to ``"string"`` for anything unrecognised.
    """
    import types
    import typing

    # Unwrap Optional[X] / Union[X, None]
    origin = getattr(annotation, "__origin__", None)
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in annotation.__args__ if a is not type(None)]
        if args:
            return _python_type_to_json_type(args[0])

    return _PYTHON_TYPE_TO_JSON.get(annotation, "string")


def _build_input_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Build an Anthropic-compatible ``input_schema`` dict from *func*'s signature."""
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "return":
            continue

        annotation = hints.get(param_name, str)
        json_type = _python_type_to_json_type(annotation)

        prop: dict[str, Any] = {"type": json_type}

        # Pull per-parameter description from docstring if present
        # (kept simple: no dependency on docstring_parser here)
        properties[param_name] = prop

        # A parameter is required when it has no default value
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Registry that maps tool names to callables and their JSON schemas.

    Usage::

        registry = ToolRegistry()

        @registry.tool(description="Read a file and return its contents")
        def read_file(path: str) -> str:
            with open(path) as f:
                return f.read()

        # Produce the list of tool dicts to pass to the Anthropic API
        schema = registry.to_api_schema()

        # Dispatch an LLM tool-use block
        result = registry.execute("read_file", {"path": "/etc/hostname"})
    """

    def __init__(self) -> None:
        # Ordered mapping of tool name → {"func": ..., "description": ..., "schema": ...}
        self._tools: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def tool(
        self,
        description: str,
        *,
        name: str | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator factory that registers *func* as a named tool.

        Args:
            description: Human-readable description sent to the LLM.
            name: Override the tool name (defaults to ``func.__name__``).
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or func.__name__
            self._tools[tool_name] = {
                "func": func,
                "description": description,
                "input_schema": _build_input_schema(func),
            }
            return func

        return decorator

    def register(
        self,
        func: Callable[..., Any],
        description: str,
        *,
        name: str | None = None,
    ) -> None:
        """Programmatic (non-decorator) registration of a tool.

        Args:
            func: The callable to register.
            description: Human-readable description.
            name: Override the tool name (defaults to ``func.__name__``).
        """
        tool_name = name or func.__name__
        self._tools[tool_name] = {
            "func": func,
            "description": description,
            "input_schema": _build_input_schema(func),
        }

    # ------------------------------------------------------------------
    # Schema export
    # ------------------------------------------------------------------

    def to_api_schema(self) -> list[dict[str, Any]]:
        """Return the list of tool dicts suitable for the Anthropic ``tools=`` parameter.

        Each entry has the shape::

            {
                "name": "tool_name",
                "description": "...",
                "input_schema": {
                    "type": "object",
                    "properties": { ... },
                    "required": [ ... ],
                },
            }
        """
        return [
            {
                "name": tool_name,
                "description": entry["description"],
                "input_schema": entry["input_schema"],
            }
            for tool_name, entry in self._tools.items()
        ]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, name: str, tool_input: dict[str, Any]) -> Any:
        """Execute the tool named *name* with the provided *tool_input* dict.

        Args:
            name: The registered tool name.
            tool_input: Keyword arguments to pass to the tool function.

        Returns:
            Whatever the tool function returns.

        Raises:
            ToolNotFoundError: If *name* is not registered.
        """
        if name not in self._tools:
            raise ToolNotFoundError(name)

        func = self._tools[name]["func"]
        return func(**tool_input)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        """Return the list of registered tool names."""
        return list(self._tools.keys())
