"""Core agent loop: the raw while-not-done mechanism."""

from __future__ import annotations

from typing import Any

import anthropic

from agent.logger import RunLogger
from agent.tool_registry import ToolRegistry

# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def _tool_result_message(tool_use_id: str, content: str) -> dict[str, Any]:
    """Build a ``tool_result`` user message for the Anthropic API."""
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": str(content),
            }
        ],
    }


def _extract_text(response: anthropic.types.Message) -> str:
    """Extract the first text block from a response, or return an empty string."""
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


def _response_cost(response: anthropic.types.Message) -> float:
    """Estimate USD cost from token usage.

    Uses Claude 3.5 Haiku pricing as a reasonable default
    ($0.80/M input, $4.00/M output).  The caller can override by
    passing an explicit ``cost_usd`` when logging.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0.0
    try:
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    return (input_tokens * 0.80 + output_tokens * 4.00) / 1_000_000


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    """Minimal agent that drives a tool-use loop until the model stops.

    Args:
        registry: The :class:`~agent.tool_registry.ToolRegistry` holding
            the tools available to the agent.
        model: Anthropic model identifier to use.
        max_steps: Safety cap on the number of tool-call rounds.
        client: Optional pre-configured :class:`anthropic.Anthropic` client
            (useful for testing with a mock).
    """

    def __init__(
        self,
        registry: ToolRegistry,
        model: str = "claude-3-5-haiku-20241022",
        max_steps: int = 20,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.registry = registry
        self.model = model
        self.max_steps = max_steps
        self._client = client or anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, task: str) -> str:
        """Run the agent on *task* and return the final text response.

        Every LLM call and tool execution is recorded to SQLite via
        :class:`~agent.logger.RunLogger`.

        The loop:
        1. Append the user message.
        2. Call the model.
        3. If the model wants to use tools, execute each tool call and
           append the results, then repeat.
        4. When the model stops with ``end_turn`` (or any non-``tool_use``
           stop reason), extract and return the final text.

        Args:
            task: Natural-language description of the task to complete.

        Returns:
            The final text response from the model.

        Raises:
            RuntimeError: If ``max_steps`` is exceeded before the model
                produces a non-tool-use response.
        """
        run_logger = RunLogger(task=task, model=self.model)
        run_logger.open()

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": task},
        ]

        done = False
        steps = 0
        total_cost: float = 0.0
        final_output: str = ""
        end_status: str = "success"

        try:
            while not done:
                if steps >= self.max_steps:
                    end_status = "max_steps"
                    raise RuntimeError(
                        f"Agent exceeded max_steps={self.max_steps} without finishing."
                    )

                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    tools=self.registry.to_api_schema(),
                    messages=messages,
                )

                # Calculate token usage and cost for this LLM call
                usage = getattr(response, "usage", None)
                _raw_in = getattr(usage, "input_tokens", None)
                _raw_out = getattr(usage, "output_tokens", None)
                try:
                    input_tokens: int | None = int(_raw_in) if _raw_in is not None else None
                except (TypeError, ValueError):
                    input_tokens = None
                try:
                    output_tokens: int | None = int(_raw_out) if _raw_out is not None else None
                except (TypeError, ValueError):
                    output_tokens = None
                step_cost = _response_cost(response)
                total_cost += step_cost

                # Log the LLM call
                run_logger.log_step(
                    type="llm_call",
                    tool_output=_extract_text(response) or None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=step_cost,
                )

                # Append the assistant turn to the conversation history
                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "tool_use":
                    # Execute every tool call in this response and collect results
                    tool_result_contents: list[dict[str, Any]] = []

                    for block in response.content:
                        if block.type != "tool_use":
                            continue

                        result = self.registry.execute(block.name, block.input)

                        # Log each tool execution as its own step
                        run_logger.log_step(
                            type="tool_use",
                            tool_name=block.name,
                            tool_input=dict(block.input),
                            tool_output=str(result),
                        )

                        tool_result_contents.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": str(result),
                            }
                        )

                    # All tool results go back in a single user message
                    messages.append(
                        {"role": "user", "content": tool_result_contents}
                    )

                else:
                    done = True
                    final_output = _extract_text(response)

                steps += 1

        except Exception:
            if end_status == "success":
                end_status = "failed"
            run_logger.finish(
                status=end_status,
                total_cost_usd=total_cost,
                final_output=final_output or None,
            )
            run_logger.close()
            raise

        run_logger.finish(
            status=end_status,
            total_cost_usd=total_cost,
            final_output=final_output or None,
        )
        run_logger.close()

        return final_output
