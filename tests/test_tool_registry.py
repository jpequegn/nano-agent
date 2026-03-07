"""Tests for ToolRegistry — acceptance criteria for issue #3.

Acceptance criteria:
  - Register 3 tools
  - Serialize schema (to_api_schema)
  - Execute each tool successfully
  - Raise ToolNotFoundError for unknown tools
"""

from __future__ import annotations

import pytest

from agent.tool_registry import ToolNotFoundError, ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> ToolRegistry:
    """Return a fresh ToolRegistry with three tools registered."""
    reg = ToolRegistry()

    @reg.tool(description="Return the contents of a file at *path*.")
    def read_file(path: str) -> str:
        with open(path) as fh:
            return fh.read()

    @reg.tool(description="Add two integers and return the result.")
    def add(x: int, y: int) -> int:
        return x + y

    @reg.tool(description="Greet a person by name, optionally with a custom greeting.")
    def greet(name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}!"

    return reg


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_three_tools_registered(self, registry: ToolRegistry) -> None:
        assert len(registry) == 3

    def test_tool_names(self, registry: ToolRegistry) -> None:
        assert set(registry.names()) == {"read_file", "add", "greet"}

    def test_contains(self, registry: ToolRegistry) -> None:
        assert "read_file" in registry
        assert "add" in registry
        assert "greet" in registry
        assert "unknown" not in registry

    def test_register_programmatic(self) -> None:
        """ToolRegistry.register() (non-decorator) works correctly."""
        reg = ToolRegistry()

        def multiply(a: int, b: int) -> int:
            return a * b

        reg.register(multiply, description="Multiply two integers.")
        assert "multiply" in reg

    def test_register_with_name_override(self) -> None:
        reg = ToolRegistry()

        @reg.tool(description="Echo the input.", name="echo_tool")
        def _echo(text: str) -> str:
            return text

        assert "echo_tool" in reg
        assert "_echo" not in reg


# ---------------------------------------------------------------------------
# Schema serialisation
# ---------------------------------------------------------------------------


class TestSchema:
    def test_to_api_schema_returns_list(self, registry: ToolRegistry) -> None:
        schema = registry.to_api_schema()
        assert isinstance(schema, list)
        assert len(schema) == 3

    def test_schema_top_level_keys(self, registry: ToolRegistry) -> None:
        schema = registry.to_api_schema()
        for entry in schema:
            assert "name" in entry
            assert "description" in entry
            assert "input_schema" in entry

    def test_read_file_schema(self, registry: ToolRegistry) -> None:
        schema = registry.to_api_schema()
        entry = next(e for e in schema if e["name"] == "read_file")

        assert entry["description"] == "Return the contents of a file at *path*."
        assert entry["input_schema"]["type"] == "object"
        props = entry["input_schema"]["properties"]
        assert "path" in props
        assert props["path"]["type"] == "string"
        assert entry["input_schema"]["required"] == ["path"]

    def test_add_schema(self, registry: ToolRegistry) -> None:
        schema = registry.to_api_schema()
        entry = next(e for e in schema if e["name"] == "add")

        props = entry["input_schema"]["properties"]
        assert props["x"]["type"] == "integer"
        assert props["y"]["type"] == "integer"
        assert set(entry["input_schema"]["required"]) == {"x", "y"}

    def test_greet_optional_param_not_required(self, registry: ToolRegistry) -> None:
        schema = registry.to_api_schema()
        entry = next(e for e in schema if e["name"] == "greet")

        required = entry["input_schema"].get("required", [])
        assert "name" in required
        assert "greeting" not in required  # has a default value


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class TestExecution:
    def test_execute_add(self, registry: ToolRegistry) -> None:
        result = registry.execute("add", {"x": 3, "y": 4})
        assert result == 7

    def test_execute_greet_default(self, registry: ToolRegistry) -> None:
        result = registry.execute("greet", {"name": "World"})
        assert result == "Hello, World!"

    def test_execute_greet_custom(self, registry: ToolRegistry) -> None:
        result = registry.execute("greet", {"name": "Claude", "greeting": "Hi"})
        assert result == "Hi, Claude!"

    def test_execute_read_file(self, tmp_path, registry: ToolRegistry) -> None:
        target = tmp_path / "hello.txt"
        target.write_text("nano-agent rocks")
        result = registry.execute("read_file", {"path": str(target)})
        assert result == "nano-agent rocks"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unknown_tool_raises(self, registry: ToolRegistry) -> None:
        with pytest.raises(ToolNotFoundError) as exc_info:
            registry.execute("does_not_exist", {})

        assert exc_info.value.name == "does_not_exist"
        assert "does_not_exist" in str(exc_info.value)

    def test_tool_not_found_error_is_key_error(self) -> None:
        """ToolNotFoundError must be a subclass of KeyError for dict-like semantics."""
        assert issubclass(ToolNotFoundError, KeyError)

    def test_bad_args_raises_type_error(self, registry: ToolRegistry) -> None:
        """Passing wrong kwargs propagates naturally as a TypeError."""
        with pytest.raises(TypeError):
            registry.execute("add", {"x": 1})  # missing y
