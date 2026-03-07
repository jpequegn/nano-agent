"""Tests for Agent — acceptance criteria for issue #6.

Acceptance criteria:
  - Agent class with run(task: str) -> str
  - Message list management (user → assistant → tool_result cycle)
  - Handle multiple tool calls in a single response
  - Return final text response
  - Agent completes a 3-step task using 2 different tools
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.agent import Agent, _extract_text, _tool_result_message
from agent.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers to build fake Anthropic response objects
# ---------------------------------------------------------------------------


def _make_text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(id: str, name: str, input: dict[str, Any]) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = id
    block.name = name
    block.input = input
    return block


def _make_response(
    stop_reason: str,
    content: list[MagicMock],
) -> MagicMock:
    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = content
    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> ToolRegistry:
    """A registry with two simple tools."""
    reg = ToolRegistry()

    @reg.tool(description="Add two integers.")
    def add(x: int, y: int) -> int:
        return x + y

    @reg.tool(description="Multiply two integers.")
    def multiply(x: int, y: int) -> int:
        return x * y

    return reg


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_tool_result_message_shape(self) -> None:
        msg = _tool_result_message("id-123", "42")
        assert msg["role"] == "user"
        assert len(msg["content"]) == 1
        block = msg["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "id-123"
        assert block["content"] == "42"

    def test_extract_text_returns_first_text_block(self) -> None:
        response = _make_response("end_turn", [_make_text_block("hello")])
        assert _extract_text(response) == "hello"

    def test_extract_text_returns_empty_when_no_text(self) -> None:
        response = _make_response("end_turn", [])
        assert _extract_text(response) == ""


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------


class TestAgentInit:
    def test_default_attributes(self, registry: ToolRegistry) -> None:
        agent = Agent(registry=registry)
        assert agent.model == "claude-3-5-haiku-20241022"
        assert agent.max_steps == 20
        assert agent.registry is registry

    def test_custom_model_and_steps(self, registry: ToolRegistry) -> None:
        agent = Agent(registry=registry, model="claude-3-opus-20240229", max_steps=5)
        assert agent.model == "claude-3-opus-20240229"
        assert agent.max_steps == 5

    def test_custom_client_is_used(self, registry: ToolRegistry) -> None:
        mock_client = MagicMock()
        agent = Agent(registry=registry, client=mock_client)
        assert agent._client is mock_client


# ---------------------------------------------------------------------------
# Core loop — single tool call then done
# ---------------------------------------------------------------------------


class TestAgentRun:
    def test_returns_final_text_no_tools(self, registry: ToolRegistry) -> None:
        """When the model returns end_turn immediately, run() returns the text."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_response(
            "end_turn", [_make_text_block("The answer is 42.")]
        )

        agent = Agent(registry=registry, client=mock_client)
        result = agent.run("What is 6 × 7?")

        assert result == "The answer is 42."

    def test_single_tool_call_cycle(self, registry: ToolRegistry) -> None:
        """One tool call followed by a final text response."""
        mock_client = MagicMock()
        tool_response = _make_response(
            "tool_use",
            [_make_tool_use_block("tid-1", "add", {"x": 3, "y": 4})],
        )
        final_response = _make_response(
            "end_turn", [_make_text_block("3 + 4 = 7")]
        )
        mock_client.messages.create.side_effect = [tool_response, final_response]

        agent = Agent(registry=registry, client=mock_client)
        result = agent.run("Add 3 and 4.")

        assert result == "3 + 4 = 7"
        assert mock_client.messages.create.call_count == 2

    def test_message_history_is_built_correctly(self, registry: ToolRegistry) -> None:
        """Verify the message list: user → assistant → tool_result → assistant.

        Because the messages list is mutated in-place, the second call's
        kwargs["messages"] already shows the full 4-message history.
        """
        mock_client = MagicMock()
        tool_response = _make_response(
            "tool_use",
            [_make_tool_use_block("tid-1", "add", {"x": 1, "y": 2})],
        )
        final_response = _make_response(
            "end_turn", [_make_text_block("Done")]
        )
        mock_client.messages.create.side_effect = [tool_response, final_response]

        agent = Agent(registry=registry, client=mock_client)
        agent.run("task")

        # The shared messages list after the loop:
        # [0] user (task)
        # [1] assistant (tool_use)
        # [2] user (tool_result)
        # [3] assistant (text/end_turn)
        final_messages = mock_client.messages.create.call_args_list[1].kwargs[
            "messages"
        ]
        assert final_messages[0]["role"] == "user"
        assert final_messages[1]["role"] == "assistant"
        assert final_messages[2]["role"] == "user"
        # The third message must contain a tool_result block
        content = final_messages[2]["content"]
        assert any(b["type"] == "tool_result" for b in content)

    # ------------------------------------------------------------------
    # Acceptance criteria: 3-step task using 2 different tools
    # ------------------------------------------------------------------

    def test_three_step_task_two_tools(self, registry: ToolRegistry) -> None:
        """Agent completes a 3-step task using 2 different tools.

        Step 1: model calls `add(x=3, y=4)`      → result 7
        Step 2: model calls `multiply(x=7, y=6)`  → result 42
        Step 3: model returns final text

        The messages list is mutated in-place, so by the time the third call
        is made, all 6 messages are present.  We verify the cumulative history
        rather than per-call snapshots.
        """
        mock_client = MagicMock()

        step1 = _make_response(
            "tool_use",
            [_make_tool_use_block("tid-1", "add", {"x": 3, "y": 4})],
        )
        step2 = _make_response(
            "tool_use",
            [_make_tool_use_block("tid-2", "multiply", {"x": 7, "y": 6})],
        )
        step3 = _make_response(
            "end_turn",
            [_make_text_block("(3 + 4) × 6 = 42")],
        )
        mock_client.messages.create.side_effect = [step1, step2, step3]

        agent = Agent(registry=registry, client=mock_client)
        result = agent.run("Compute (3 + 4) then multiply the result by 6.")

        assert result == "(3 + 4) × 6 = 42"
        # Three API calls were made (one per step)
        assert mock_client.messages.create.call_count == 3

        # By the final call the full conversation history is present.
        # Layout: [user, assistant(add), tool_result(7), assistant(multiply), tool_result(42), assistant(text)]
        # The messages list is shared (mutated in-place), so all call_args show the same list.
        final_msgs = mock_client.messages.create.call_args_list[2].kwargs["messages"]

        # Positions 1 and 2: first tool call + its result
        assert final_msgs[1]["role"] == "assistant"
        first_tool_result = final_msgs[2]
        assert first_tool_result["role"] == "user"
        assert first_tool_result["content"][0]["type"] == "tool_result"
        assert first_tool_result["content"][0]["content"] == "7"   # 3 + 4

        # Positions 3 and 4: second tool call + its result
        assert final_msgs[3]["role"] == "assistant"
        second_tool_result = final_msgs[4]
        assert second_tool_result["role"] == "user"
        assert second_tool_result["content"][0]["type"] == "tool_result"
        assert second_tool_result["content"][0]["content"] == "42"  # 7 × 6

    # ------------------------------------------------------------------
    # Multiple tool calls in a single response
    # ------------------------------------------------------------------

    def test_multiple_tool_calls_in_single_response(self, registry: ToolRegistry) -> None:
        """Two tool calls in one response are both executed and returned together.

        Layout after the loop:
          [0] user           – original task
          [1] assistant      – parallel tool calls (add + multiply)
          [2] user           – tool_result for both (single message, two blocks)
          [3] assistant      – final text
        """
        mock_client = MagicMock()

        parallel_step = _make_response(
            "tool_use",
            [
                _make_tool_use_block("tid-a", "add", {"x": 1, "y": 2}),
                _make_tool_use_block("tid-b", "multiply", {"x": 3, "y": 4}),
            ],
        )
        final_step = _make_response(
            "end_turn",
            [_make_text_block("1+2=3 and 3×4=12")],
        )
        mock_client.messages.create.side_effect = [parallel_step, final_step]

        agent = Agent(registry=registry, client=mock_client)
        result = agent.run("Compute 1+2 and 3×4 in parallel.")

        assert result == "1+2=3 and 3×4=12"

        # The final messages list (shared, mutated in-place) has 4 entries.
        # Index 2 is the user message with both tool_result blocks.
        final_msgs = mock_client.messages.create.call_args_list[1].kwargs["messages"]
        tool_results_content = final_msgs[2]["content"]
        assert len(tool_results_content) == 2
        results_by_id = {b["tool_use_id"]: b["content"] for b in tool_results_content}
        assert results_by_id["tid-a"] == "3"   # 1 + 2
        assert results_by_id["tid-b"] == "12"  # 3 × 4

    # ------------------------------------------------------------------
    # Safety: max_steps exceeded
    # ------------------------------------------------------------------

    def test_max_steps_exceeded_raises(self, registry: ToolRegistry) -> None:
        """RuntimeError is raised when the agent exceeds max_steps."""
        mock_client = MagicMock()
        # Always return tool_use so the loop never ends naturally
        mock_client.messages.create.return_value = _make_response(
            "tool_use",
            [_make_tool_use_block("tid-inf", "add", {"x": 0, "y": 0})],
        )

        agent = Agent(registry=registry, client=mock_client, max_steps=3)
        with pytest.raises(RuntimeError, match="max_steps=3"):
            agent.run("loop forever")

    # ------------------------------------------------------------------
    # Tool names and API schema are forwarded to the model
    # ------------------------------------------------------------------

    def test_tools_schema_passed_to_api(self, registry: ToolRegistry) -> None:
        """The agent passes registry.to_api_schema() to every API call."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_response(
            "end_turn", [_make_text_block("ok")]
        )

        agent = Agent(registry=registry, client=mock_client)
        agent.run("anything")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["tools"] == registry.to_api_schema()
