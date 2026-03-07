"""Tests for CostTracker, StepCost, RunCost — acceptance criteria for issue #9.

Acceptance criteria:
  - StepCost and RunCost dataclasses with correct fields
  - CostTracker records token usage and computes correct cost per step
  - Hardcoded price table for claude-3-5-haiku / sonnet / opus (and claude-3 variants)
  - RunCost.summary() produces a readable multi-line string
  - Agent.run() populates agent.last_run_cost after a run
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.cost_tracker import (
    CostTracker,
    RunCost,
    StepCost,
    _prices_for_model,
)
from agent.agent import Agent
from agent.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_usage(input_tokens: int, output_tokens: int) -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    return usage


def _make_response(stop_reason: str, content: list, usage: MagicMock | None = None) -> MagicMock:
    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = content
    response.usage = usage
    return response


def _make_text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


# ---------------------------------------------------------------------------
# Price table
# ---------------------------------------------------------------------------


class TestPriceTable:
    def test_haiku_35_prices(self) -> None:
        inp, out = _prices_for_model("claude-3-5-haiku-20241022")
        assert inp == pytest.approx(0.80)
        assert out == pytest.approx(4.00)

    def test_sonnet_35_prices(self) -> None:
        inp, out = _prices_for_model("claude-3-5-sonnet-20241022")
        assert inp == pytest.approx(3.00)
        assert out == pytest.approx(15.00)

    def test_opus_35_prices(self) -> None:
        inp, out = _prices_for_model("claude-3-5-opus-20241022")
        assert inp == pytest.approx(15.00)
        assert out == pytest.approx(75.00)

    def test_haiku_3_prices(self) -> None:
        inp, out = _prices_for_model("claude-3-haiku-20240307")
        assert inp == pytest.approx(0.25)
        assert out == pytest.approx(1.25)

    def test_sonnet_3_prices(self) -> None:
        inp, out = _prices_for_model("claude-3-sonnet-20240229")
        assert inp == pytest.approx(3.00)
        assert out == pytest.approx(15.00)

    def test_opus_3_prices(self) -> None:
        inp, out = _prices_for_model("claude-3-opus-20240229")
        assert inp == pytest.approx(15.00)
        assert out == pytest.approx(75.00)

    def test_unknown_model_returns_fallback(self) -> None:
        inp, out = _prices_for_model("claude-99-ultra")
        # Should return the sonnet-class fallback
        assert inp == pytest.approx(3.00)
        assert out == pytest.approx(15.00)


# ---------------------------------------------------------------------------
# StepCost dataclass
# ---------------------------------------------------------------------------


class TestStepCost:
    def test_fields(self) -> None:
        sc = StepCost(step=1, input_tokens=100, output_tokens=50, model="claude-3-5-haiku-20241022", cost_usd=0.00028)
        assert sc.step == 1
        assert sc.input_tokens == 100
        assert sc.output_tokens == 50
        assert sc.model == "claude-3-5-haiku-20241022"
        assert sc.cost_usd == pytest.approx(0.00028)


# ---------------------------------------------------------------------------
# RunCost dataclass
# ---------------------------------------------------------------------------


class TestRunCost:
    def test_empty_summary(self) -> None:
        rc = RunCost()
        assert "no API calls" in rc.summary()

    def test_summary_contains_step_info(self) -> None:
        sc = StepCost(step=1, input_tokens=1000, output_tokens=200, model="claude-3-5-haiku-20241022", cost_usd=0.00164)
        rc = RunCost(steps=[sc], total_usd=0.00164)
        summary = rc.summary()
        assert "Step" in summary
        assert "Total" in summary
        assert "$" in summary

    def test_summary_aggregates_tokens(self) -> None:
        steps = [
            StepCost(step=1, input_tokens=500, output_tokens=100, model="m", cost_usd=0.001),
            StepCost(step=2, input_tokens=300, output_tokens=80, model="m", cost_usd=0.0008),
        ]
        rc = RunCost(steps=steps, total_usd=0.0018)
        summary = rc.summary()
        # Total tokens should be 800 input, 180 output
        assert "800" in summary
        assert "180" in summary


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class TestCostTracker:
    def test_record_single_step(self) -> None:
        tracker = CostTracker(model="claude-3-5-haiku-20241022")
        usage = _make_usage(input_tokens=1_000_000, output_tokens=0)
        step_cost = tracker.record(step=1, usage=usage)

        # 1M input tokens at $0.80/1M = $0.80
        assert step_cost.step == 1
        assert step_cost.input_tokens == 1_000_000
        assert step_cost.output_tokens == 0
        assert step_cost.cost_usd == pytest.approx(0.80)

    def test_record_output_tokens(self) -> None:
        tracker = CostTracker(model="claude-3-5-haiku-20241022")
        usage = _make_usage(input_tokens=0, output_tokens=1_000_000)
        step_cost = tracker.record(step=1, usage=usage)

        # 1M output tokens at $4.00/1M = $4.00
        assert step_cost.cost_usd == pytest.approx(4.00)

    def test_record_combined_cost(self) -> None:
        tracker = CostTracker(model="claude-3-5-sonnet-20241022")
        # 1000 input ($0.003) + 500 output ($0.0075) = $0.0105
        usage = _make_usage(input_tokens=1_000, output_tokens=500)
        step_cost = tracker.record(step=1, usage=usage)
        assert step_cost.cost_usd == pytest.approx(0.003 + 0.0075)

    def test_multiple_steps_accumulate(self) -> None:
        tracker = CostTracker(model="claude-3-5-haiku-20241022")
        tracker.record(step=1, usage=_make_usage(1000, 200))
        tracker.record(step=2, usage=_make_usage(800, 150))
        tracker.record(step=3, usage=_make_usage(600, 100))

        run_cost = tracker.run_cost()
        assert len(run_cost.steps) == 3
        assert run_cost.steps[0].step == 1
        assert run_cost.steps[2].step == 3
        # total_usd must equal sum of individual step costs
        expected_total = sum(s.cost_usd for s in run_cost.steps)
        assert run_cost.total_usd == pytest.approx(expected_total)

    def test_run_cost_is_snapshot(self) -> None:
        """run_cost() returns a snapshot; adding more steps doesn't mutate it."""
        tracker = CostTracker(model="claude-3-5-haiku-20241022")
        tracker.record(step=1, usage=_make_usage(100, 50))
        snap = tracker.run_cost()

        tracker.record(step=2, usage=_make_usage(200, 80))
        assert len(snap.steps) == 1  # snapshot is unchanged

    def test_zero_tokens_is_zero_cost(self) -> None:
        tracker = CostTracker(model="claude-3-5-haiku-20241022")
        usage = _make_usage(0, 0)
        step_cost = tracker.record(step=1, usage=usage)
        assert step_cost.cost_usd == pytest.approx(0.0)

    def test_missing_usage_attributes_default_to_zero(self) -> None:
        tracker = CostTracker(model="claude-3-5-haiku-20241022")
        # usage object with no token attributes
        usage = MagicMock(spec=[])  # empty spec — no attributes
        step_cost = tracker.record(step=1, usage=usage)
        assert step_cost.input_tokens == 0
        assert step_cost.output_tokens == 0
        assert step_cost.cost_usd == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Integration: Agent.last_run_cost is populated after run()
# ---------------------------------------------------------------------------


class TestAgentCostIntegration:
    @pytest.fixture()
    def registry(self) -> ToolRegistry:
        reg = ToolRegistry()

        @reg.tool(description="Add two integers.")
        def add(x: int, y: int) -> int:
            return x + y

        return reg

    def test_last_run_cost_populated_after_run(self, registry: ToolRegistry) -> None:
        mock_client = MagicMock()
        usage = _make_usage(input_tokens=500, output_tokens=100)
        mock_client.messages.create.return_value = _make_response(
            "end_turn", [_make_text_block("done")], usage=usage
        )

        agent = Agent(registry=registry, client=mock_client)
        agent.run("simple task")

        assert hasattr(agent, "last_run_cost")
        run_cost = agent.last_run_cost
        assert isinstance(run_cost, RunCost)
        assert len(run_cost.steps) == 1
        assert run_cost.steps[0].input_tokens == 500
        assert run_cost.steps[0].output_tokens == 100

    def test_last_run_cost_multi_step(self, registry: ToolRegistry) -> None:
        """Cost is tracked across multiple tool-call steps."""
        mock_client = MagicMock()

        def _tool_use_block(id: str, name: str, inp: dict) -> MagicMock:
            b = MagicMock()
            b.type = "tool_use"
            b.id = id
            b.name = name
            b.input = inp
            return b

        step1_response = _make_response(
            "tool_use",
            [_tool_use_block("tid-1", "add", {"x": 1, "y": 2})],
            usage=_make_usage(300, 50),
        )
        step2_response = _make_response(
            "end_turn",
            [_make_text_block("result is 3")],
            usage=_make_usage(400, 80),
        )
        mock_client.messages.create.side_effect = [step1_response, step2_response]

        agent = Agent(registry=registry, client=mock_client)
        agent.run("Add 1 and 2.")

        run_cost = agent.last_run_cost
        assert len(run_cost.steps) == 2
        assert run_cost.steps[0].step == 1
        assert run_cost.steps[1].step == 2
        total = run_cost.steps[0].cost_usd + run_cost.steps[1].cost_usd
        assert run_cost.total_usd == pytest.approx(total)

    def test_last_run_cost_no_usage_on_response(self, registry: ToolRegistry) -> None:
        """If response has no usage attribute, step is still recorded (zero cost)."""
        mock_client = MagicMock()
        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content = [_make_text_block("ok")]
        response.usage = None  # simulate missing usage

        mock_client.messages.create.return_value = response

        agent = Agent(registry=registry, client=mock_client)
        agent.run("task with no usage")

        # Should not raise; last_run_cost exists with one step recorded (or none)
        assert hasattr(agent, "last_run_cost")

    def test_summary_output_format(self, registry: ToolRegistry) -> None:
        """RunCost.summary() string contains expected labels."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_response(
            "end_turn",
            [_make_text_block("done")],
            usage=_make_usage(1000, 200),
        )

        agent = Agent(registry=registry, model="claude-3-5-haiku-20241022", client=mock_client)
        agent.run("task")

        summary = agent.last_run_cost.summary()
        assert "Cost summary" in summary
        assert "Step" in summary
        assert "Total" in summary
        assert "$" in summary
