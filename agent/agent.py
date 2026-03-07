"""Core agent loop: the raw while-not-done mechanism."""

from __future__ import annotations

from typing import Any

import anthropic

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
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": task},
        ]

        done = False
        steps = 0

        while not done:
            if steps >= self.max_steps:
                raise RuntimeError(
                    f"Agent exceeded max_steps={self.max_steps} without finishing."
                )

            response = self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                tools=self.registry.to_api_schema(),
                messages=messages,
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

            steps += 1

        return _extract_text(response)  # type: ignore[possibly-undefined]
