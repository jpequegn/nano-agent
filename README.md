# nano-agent

A minimal LLM coding agent built with the Anthropic SDK. Designed for learning — no frameworks, just the raw loop.

## Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Configure your API key
cp .env.example .env
# edit .env and add your ANTHROPIC_API_KEY
```

## Usage

```bash
uv run agent run "write a hello world python script"
uv run agent logs
uv run agent report
```

## Architecture

```
agent/
  cli.py        # CLI entrypoint (agent run / logs / report)
  tools.py      # Tool registry and built-in tools (coming soon)
  loop.py       # Core agent loop (coming soon)
  logger.py     # SQLite run logger (coming soon)
  cost.py       # Per-step cost tracker (coming soon)
```
