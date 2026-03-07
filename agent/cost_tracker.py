"""Per-step cost tracking for agent runs.

Records token usage and compute cost for each API call, and summarises the
total cost of a full agent run.

Price table (USD per million tokens, as of 2025-03):
  https://www.anthropic.com/pricing

Supported model families:
  claude-3-5-haiku   — fastest / cheapest 3.5-class model
  claude-3-5-sonnet  — balanced 3.5-class model
  claude-3-5-opus    — most capable 3.5-class model
  claude-3-haiku     — fastest / cheapest 3.x model
  claude-3-sonnet    — balanced 3.x model
  claude-3-opus      — most capable 3.x model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Price table  (input_usd_per_1m, output_usd_per_1m)
# ---------------------------------------------------------------------------

# Keys are *prefix* patterns matched against the model string.
# The first matching entry wins, so more-specific prefixes must come first.
_PRICE_TABLE: list[tuple[str, float, float]] = [
    # claude-3-5 family
    ("claude-3-5-haiku",  0.80,   4.00),
    ("claude-3-5-sonnet", 3.00,  15.00),
    ("claude-3-5-opus",  15.00,  75.00),
    # claude-3 family
    ("claude-3-haiku",    0.25,   1.25),
    ("claude-3-sonnet",   3.00,  15.00),
    ("claude-3-opus",    15.00,  75.00),
    # claude-3-7 family
    ("claude-3-7-sonnet", 3.00,  15.00),
]

_FALLBACK_PRICES: tuple[float, float] = (3.00, 15.00)  # sonnet-class defaults


def _prices_for_model(model: str) -> tuple[float, float]:
    """Return *(input_usd_per_1m, output_usd_per_1m)* for *model*.

    Matches on a prefix basis so that date-suffixed model IDs
    (e.g. ``claude-3-5-haiku-20241022``) resolve correctly.
    """
    for prefix, input_price, output_price in _PRICE_TABLE:
        if model.startswith(prefix):
            return input_price, output_price
    return _FALLBACK_PRICES


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StepCost:
    """Cost record for a single API call (one agent step).

    Attributes:
        step: 1-based step index within the run.
        input_tokens: Number of input (prompt) tokens used.
        output_tokens: Number of output (completion) tokens used.
        model: Model identifier used for this call.
        cost_usd: Total cost in USD for this step.
    """

    step: int
    input_tokens: int
    output_tokens: int
    model: str
    cost_usd: float

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"  Step {self.step:>2}: "
            f"in={self.input_tokens:>6} out={self.output_tokens:>5} "
            f"model={self.model}  "
            f"cost=${self.cost_usd:.6f}"
        )


@dataclass
class RunCost:
    """Aggregated cost for a complete agent run.

    Attributes:
        steps: Ordered list of per-step cost records.
        total_usd: Sum of all step costs.
    """

    steps: list[StepCost] = field(default_factory=list)
    total_usd: float = 0.0

    def summary(self) -> str:
        """Return a human-readable multi-line cost summary."""
        if not self.steps:
            return "Cost summary: no API calls recorded."

        lines: list[str] = ["Cost summary:"]
        lines.extend(str(step) for step in self.steps)

        total_in = sum(s.input_tokens for s in self.steps)
        total_out = sum(s.output_tokens for s in self.steps)
        lines.append(
            f"  Total : "
            f"in={total_in:>6} out={total_out:>5} "
            f"steps={len(self.steps)}  "
            f"cost=${self.total_usd:.6f}"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class CostTracker:
    """Accumulates per-step costs across a single agent run.

    Usage::

        tracker = CostTracker(model="claude-3-5-haiku-20241022")
        tracker.record(step=1, usage=response.usage)
        ...
        run_cost = tracker.run_cost()
        print(run_cost.summary())
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self._steps: list[StepCost] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, step: int, usage: Any) -> StepCost:
        """Record the token usage from one API response.

        Args:
            step: 1-based step index.
            usage: The ``usage`` object from an
                :class:`anthropic.types.Message` response.  Must expose
                ``input_tokens`` and ``output_tokens`` attributes.

        Returns:
            The :class:`StepCost` that was recorded.
        """
        input_tokens: int = getattr(usage, "input_tokens", 0) or 0
        output_tokens: int = getattr(usage, "output_tokens", 0) or 0

        input_price, output_price = _prices_for_model(self.model)

        cost_usd = (
            input_tokens * input_price / 1_000_000
            + output_tokens * output_price / 1_000_000
        )

        step_cost = StepCost(
            step=step,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            cost_usd=cost_usd,
        )
        self._steps.append(step_cost)
        return step_cost

    def run_cost(self) -> RunCost:
        """Return the aggregated :class:`RunCost` for this run."""
        return RunCost(
            steps=list(self._steps),
            total_usd=sum(s.cost_usd for s in self._steps),
        )
