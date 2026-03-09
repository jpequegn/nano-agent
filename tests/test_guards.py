"""Tests for max-steps guard and wall-clock timeout — acceptance criteria for issue #1.

Acceptance criteria:
  - MaxStepsExceeded raised (not bare RuntimeError) when max_steps is hit
  - RunTimeout raised when timeout_seconds elapses
  - Both exceptions carry partial_output and the relevant limit attribute
  - Both statuses ('max_steps', 'timeout') are written to the run log
  - Agent given an impossible task hits max_steps and exits cleanly
  - timeout_seconds=None disables the timeout
  - Default values: max_steps=20, timeout_seconds=120
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.agent import Agent
from agent.exceptions import MaxStepsExceeded, RunTimeout
from agent.logger import RunLogger, list_recent_runs
from agent.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(id: str, name: str, input: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = id
    block.name = name
    block.input = input
    return block


def _make_response(stop_reason: str, content: list) -> MagicMock:
    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = content
    response.usage = None
    return response


@pytest.fixture()
def registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(description="No-op tool.")
    def noop() -> str:
        return "noop"

    return reg


# ---------------------------------------------------------------------------
# Exception dataclasses
# ---------------------------------------------------------------------------


class TestMaxStepsExceeded:
    def test_message_contains_limit(self) -> None:
        exc = MaxStepsExceeded(max_steps=5)
        assert "5" in str(exc)

    def test_max_steps_attribute(self) -> None:
        exc = MaxStepsExceeded(max_steps=10)
        assert exc.max_steps == 10

    def test_partial_output_default_empty(self) -> None:
        exc = MaxStepsExceeded(max_steps=5)
        assert exc.partial_output == ""

    def test_partial_output_stored(self) -> None:
        exc = MaxStepsExceeded(max_steps=5, partial_output="so far so good")
        assert exc.partial_output == "so far so good"

    def test_is_runtime_error(self) -> None:
        assert isinstance(MaxStepsExceeded(max_steps=1), RuntimeError)


class TestRunTimeout:
    def test_message_contains_limit(self) -> None:
        exc = RunTimeout(timeout_seconds=30.0)
        assert "30.0" in str(exc)

    def test_timeout_seconds_attribute(self) -> None:
        exc = RunTimeout(timeout_seconds=60.0)
        assert exc.timeout_seconds == 60.0

    def test_partial_output_default_empty(self) -> None:
        exc = RunTimeout(timeout_seconds=10.0)
        assert exc.partial_output == ""

    def test_partial_output_stored(self) -> None:
        exc = RunTimeout(timeout_seconds=10.0, partial_output="partial")
        assert exc.partial_output == "partial"

    def test_is_timeout_error(self) -> None:
        assert isinstance(RunTimeout(timeout_seconds=1.0), TimeoutError)


# ---------------------------------------------------------------------------
# Agent defaults
# ---------------------------------------------------------------------------


class TestAgentDefaults:
    def test_default_max_steps(self, registry: ToolRegistry) -> None:
        agent = Agent(registry=registry)
        assert agent.max_steps == 20

    def test_default_timeout_seconds(self, registry: ToolRegistry) -> None:
        agent = Agent(registry=registry)
        assert agent.timeout_seconds == 120.0

    def test_custom_max_steps(self, registry: ToolRegistry) -> None:
        agent = Agent(registry=registry, max_steps=5)
        assert agent.max_steps == 5

    def test_custom_timeout_seconds(self, registry: ToolRegistry) -> None:
        agent = Agent(registry=registry, timeout_seconds=30.0)
        assert agent.timeout_seconds == 30.0

    def test_timeout_none_disables(self, registry: ToolRegistry) -> None:
        agent = Agent(registry=registry, timeout_seconds=None)
        assert agent.timeout_seconds is None


# ---------------------------------------------------------------------------
# MaxStepsExceeded — raised and logged
# ---------------------------------------------------------------------------


class TestMaxStepsGuard:
    def test_raises_max_steps_exceeded_not_runtime_error(self, registry: ToolRegistry) -> None:
        """Must raise MaxStepsExceeded, not a bare RuntimeError."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_response(
            "tool_use",
            [_make_tool_use_block("t1", "noop", {})],
        )

        agent = Agent(registry=registry, client=mock_client, max_steps=2, timeout_seconds=None)
        with pytest.raises(MaxStepsExceeded):
            agent.run("impossible task")

    def test_max_steps_exception_carries_limit(self, registry: ToolRegistry) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_response(
            "tool_use",
            [_make_tool_use_block("t1", "noop", {})],
        )

        agent = Agent(registry=registry, client=mock_client, max_steps=3, timeout_seconds=None)
        with pytest.raises(MaxStepsExceeded) as exc_info:
            agent.run("impossible")
        assert exc_info.value.max_steps == 3

    def test_max_steps_exact_boundary(self, registry: ToolRegistry) -> None:
        """Agent with max_steps=1 completes exactly 1 LLM call then raises."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_response(
            "tool_use",
            [_make_tool_use_block("t1", "noop", {})],
        )

        agent = Agent(registry=registry, client=mock_client, max_steps=1, timeout_seconds=None)
        with pytest.raises(MaxStepsExceeded) as exc_info:
            agent.run("loop")
        assert exc_info.value.max_steps == 1
        # Exactly 1 LLM call was made before the guard fired on the next iteration
        assert mock_client.messages.create.call_count == 1

    def test_max_steps_logged_as_max_steps_status(self, registry: ToolRegistry, tmp_path: Path) -> None:
        """The run log must record status='max_steps' when the guard fires."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_response(
            "tool_use",
            [_make_tool_use_block("t1", "noop", {})],
        )
        db_path = tmp_path / "runs.db"

        # Patch the DB path used by RunLogger inside Agent
        with patch("agent.logger.DB_PATH", db_path), \
             patch("agent.agent.RunLogger") as MockLogger:
            mock_log = MagicMock()
            MockLogger.return_value = mock_log

            agent = Agent(registry=registry, client=mock_client, max_steps=2, timeout_seconds=None)
            with pytest.raises(MaxStepsExceeded):
                agent.run("loop forever")

            # finish() must have been called with status='max_steps'
            finish_calls = mock_log.finish.call_args_list
            assert len(finish_calls) == 1
            assert finish_calls[0].kwargs["status"] == "max_steps"

    def test_max_steps_partial_output_in_exception(self, registry: ToolRegistry) -> None:
        """partial_output on the exception should be empty (no text produced yet)."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_response(
            "tool_use",
            [_make_tool_use_block("t1", "noop", {})],
        )

        agent = Agent(registry=registry, client=mock_client, max_steps=1, timeout_seconds=None)
        with pytest.raises(MaxStepsExceeded) as exc_info:
            agent.run("loop")
        assert exc_info.value.partial_output == ""

    def test_impossible_task_exits_cleanly(self, registry: ToolRegistry) -> None:
        """Acceptance criterion: agent given an impossible task hits max_steps and exits cleanly."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_response(
            "tool_use",
            [_make_tool_use_block("t1", "noop", {})],
        )

        agent = Agent(registry=registry, client=mock_client, max_steps=5, timeout_seconds=None)
        with pytest.raises(MaxStepsExceeded) as exc_info:
            agent.run("do the impossible")

        # Exception is well-formed — no stray RuntimeError, no infinite loop
        assert exc_info.value.max_steps == 5
        assert mock_client.messages.create.call_count == 5


# ---------------------------------------------------------------------------
# RunTimeout — raised and logged
# ---------------------------------------------------------------------------


class TestRunTimeoutGuard:
    def test_raises_run_timeout(self, registry: ToolRegistry) -> None:
        """Must raise RunTimeout when the clock expires."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_response(
            "tool_use",
            [_make_tool_use_block("t1", "noop", {})],
        )

        # Use a very small timeout and fake time so the test is instant
        with patch("agent.agent.time") as mock_time:
            # monotonic() advances by 200 s on each call so timeout fires immediately
            mock_time.monotonic.side_effect = [0.0, 200.0, 200.0]

            agent = Agent(registry=registry, client=mock_client, max_steps=100, timeout_seconds=120.0)
            with pytest.raises(RunTimeout):
                agent.run("slow task")

    def test_run_timeout_carries_limit(self, registry: ToolRegistry) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_response(
            "tool_use",
            [_make_tool_use_block("t1", "noop", {})],
        )

        with patch("agent.agent.time") as mock_time:
            mock_time.monotonic.side_effect = [0.0, 200.0, 200.0]

            agent = Agent(registry=registry, client=mock_client, max_steps=100, timeout_seconds=60.0)
            with pytest.raises(RunTimeout) as exc_info:
                agent.run("slow")
            assert exc_info.value.timeout_seconds == 60.0

    def test_run_timeout_logged_as_timeout_status(self, registry: ToolRegistry) -> None:
        """The run log must record status='timeout' when the guard fires."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_response(
            "tool_use",
            [_make_tool_use_block("t1", "noop", {})],
        )

        with patch("agent.agent.time") as mock_time, \
             patch("agent.agent.RunLogger") as MockLogger:
            mock_time.monotonic.side_effect = [0.0, 200.0, 200.0]
            mock_log = MagicMock()
            MockLogger.return_value = mock_log

            agent = Agent(registry=registry, client=mock_client, max_steps=100, timeout_seconds=120.0)
            with pytest.raises(RunTimeout):
                agent.run("slow")

            finish_calls = mock_log.finish.call_args_list
            assert len(finish_calls) == 1
            assert finish_calls[0].kwargs["status"] == "timeout"

    def test_timeout_none_never_fires(self, registry: ToolRegistry) -> None:
        """When timeout_seconds=None, even a very long-running task won't timeout."""
        mock_client = MagicMock()
        # Immediately returns end_turn, so the loop exits naturally
        mock_client.messages.create.return_value = _make_response(
            "end_turn",
            [_make_text_block("done")],
        )

        # Even if time advances enormously, no RunTimeout should be raised
        with patch("agent.agent.time") as mock_time:
            mock_time.monotonic.return_value = 999_999.0

            agent = Agent(registry=registry, client=mock_client, timeout_seconds=None)
            result = agent.run("quick task")

        assert result == "done"

    def test_timeout_not_fired_on_fast_run(self, registry: ToolRegistry) -> None:
        """A run that completes within the timeout must not raise RunTimeout."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_response(
            "end_turn",
            [_make_text_block("fast!")],
        )

        with patch("agent.agent.time") as mock_time:
            # elapsed = 1 s, timeout = 120 s — well within limit
            mock_time.monotonic.side_effect = [0.0, 1.0]

            agent = Agent(registry=registry, client=mock_client, timeout_seconds=120.0)
            result = agent.run("fast task")

        assert result == "fast!"

    def test_timeout_partial_output_in_exception(self, registry: ToolRegistry) -> None:
        """partial_output on RunTimeout is empty when no text has been produced."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_response(
            "tool_use",
            [_make_tool_use_block("t1", "noop", {})],
        )

        with patch("agent.agent.time") as mock_time:
            mock_time.monotonic.side_effect = [0.0, 200.0, 200.0]

            agent = Agent(registry=registry, client=mock_client, timeout_seconds=120.0)
            with pytest.raises(RunTimeout) as exc_info:
                agent.run("slow")
        assert exc_info.value.partial_output == ""


# ---------------------------------------------------------------------------
# Exports from top-level package
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_max_steps_exceeded_importable(self) -> None:
        from agent import MaxStepsExceeded as MSE  # noqa: F401

        assert MSE is MaxStepsExceeded

    def test_run_timeout_importable(self) -> None:
        from agent import RunTimeout as RT  # noqa: F401

        assert RT is RunTimeout
