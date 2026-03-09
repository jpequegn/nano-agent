"""Custom exceptions for nano-agent."""

from __future__ import annotations


class MaxStepsExceeded(RuntimeError):
    """Raised when the agent exceeds its ``max_steps`` limit.

    Attributes:
        max_steps: The limit that was exceeded.
        partial_output: Any text output collected before the limit was hit
            (may be an empty string if no text was produced yet).
    """

    def __init__(self, max_steps: int, partial_output: str = "") -> None:
        self.max_steps = max_steps
        self.partial_output = partial_output
        super().__init__(
            f"Agent exceeded max_steps={max_steps} without finishing."
        )


class RunTimeout(TimeoutError):
    """Raised when the agent exceeds its ``timeout_seconds`` wall-clock limit.

    Attributes:
        timeout_seconds: The timeout that was exceeded.
        partial_output: Any text output collected before the timeout fired
            (may be an empty string if no text was produced yet).
    """

    def __init__(self, timeout_seconds: float, partial_output: str = "") -> None:
        self.timeout_seconds = timeout_seconds
        self.partial_output = partial_output
        super().__init__(
            f"Agent run exceeded timeout_seconds={timeout_seconds}."
        )
