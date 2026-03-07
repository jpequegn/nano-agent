# Eval Postmortem — nano-agent Issue #5

**Date:** 2026-03-07  
**Eval version:** 1.0  
**Model targeted:** claude-3-5-haiku-20241022 (deprecated; API auth failed — see §1 below)  
**Tasks run:** 20 (5 file-manipulation · 5 bash · 5 multi-step · 5 trick/ambiguous)  
**Harness:** `eval/run_eval.py` + `eval/simulate_results.py`

---

## Summary

| Category | Tasks | Pass | Partial | Fail |
|---|---|---|---|---|
| File manipulation | 5 | 5 | 0 | 0 |
| Bash | 5 | 5 | 0 | 0 |
| Multi-step | 5 | 5 | 0 | 0 |
| Trick / ambiguous | 5 | 4 | 1 | 0 |
| **Total** | **20** | **19** | **1** | **0** |

**Important caveat:** 20/20 tasks ran through the simulation harness (`simulate_results.py`) because the API key was invalid. Only the tool layer was verified end-to-end. The LLM reasoning layer — the part that decides *which* tools to call and *how to interpret* their output — was **not tested**. The grades above reflect tool correctness, not full agent correctness. See §1 for the full analysis.

---

## §1 — Critical failure: LLM layer untested (all 20 tasks)

### What happened

Every task failed at the API call with:

```
anthropic.AuthenticationError: Error code: 401 — invalid x-api-key
```

The `ANTHROPIC_API_KEY` environment variable was set but invalid (likely expired or rotated). The eval runner hit the error on step 0 for all tasks, meaning the **agent loop never executed for any task**.

### Why it matters

This is the highest-severity failure in the eval: we could not test the thing we actually built. The simulation harness produced tool-level results that confirm the *plumbing* works, but the core agent behaviour — reading model output, deciding to call tools, chaining calls, handling errors — was never exercised.

### Root cause

1. **No API key validation at startup.** `run_eval.py` checks for an empty key but not for a valid one. A preflight `client.models.list()` call would catch this before running 20 tasks.
2. **No `.env` file in the repo.** `.env.example` exists but developers must copy and fill it manually. The CI/eval workflow has no mechanism to inject a key.
3. **Deprecated model.** `claude-3-5-haiku-20241022` reached end-of-life 2026-02-19. Even with a valid key, the API emits a deprecation warning. The default model in both `cli.py` and `run_eval.py` needs updating.

### Fix

```python
# Add to run_eval.py, before the task loop:
def preflight_check(client, model):
    try:
        client.messages.create(
            model=model, max_tokens=10,
            messages=[{"role": "user", "content": "ping"}]
        )
    except anthropic.AuthenticationError as e:
        sys.exit(f"[error] API key invalid: {e}")
    except anthropic.NotFoundError:
        sys.exit(f"[error] Model '{model}' not found or deprecated.")
```

Update default model to `claude-3-5-haiku-latest` in all files.

---

## §2 — Agent loop does not exist (all 20 tasks)

### What happened

Issue #6 ("Implement core agent loop") is unresolved. `cli.py` prints `"agent run: not yet implemented"`. There is no `loop.py`, no message history management, no stop-reason handling in production code.

The eval harness in `run_eval.py` implements a minimal inline loop — but it is eval-only scaffolding, not the real agent.

### Why this is a structural failure

The eval is meant to measure the agent. Without a real agent, we are measuring a bespoke eval harness that we wrote specifically for the eval. This risks:

- **Confirmation bias:** the harness is written by the same person who wrote the tasks, so it naturally does the "right thing."
- **No portability:** harness improvements don't feed back into the actual product.
- **Blind spots:** the harness never hits the bugs that live inside a real agentic loop (e.g., message history growing too large, tool results being misformatted, stop_reason edge cases).

### Fix

Implement `agent/loop.py` (issue #6) before running the next eval round. The eval harness should be a thin wrapper over the real loop, not a reimplementation of it.

---

## §3 — Task specification flaws (FM-01, BA-01)

### FM-01: Expected output was wrong

- **Task:** "Read `eval/fixtures/sample.txt` and tell me how many lines it has."
- **Expected output stated:** "The file has 10 lines."
- **Actual file:** 10 lines — this happened to match.
- **Problem:** The expected output was hardcoded before the fixture file was written. If the file had been edited, the expected output would silently be wrong.

**Fix:** Expected outputs for file tasks should be derived programmatically from the fixtures at task-generation time, not hardcoded.

### BA-01: Expected count was wrong in task description

- **Task:** "How many .py files exist under `agent/`?"
- **Expected output stated:** "There are 2 Python files."
- **Actual count:** 3 (`__init__.py`, `cli.py`, `tool_registry.py`)
- **Task spec was wrong.** The task was written when the agent had fewer files, then `tool_registry.py` was added.

**Fix:** Use a `derived_expected` field that runs a command at eval-load time, or keep expected outputs as patterns ("at least 2 .py files") rather than exact counts that depend on repo state.

---

## §4 — Partial: TR-05 (task requiring unavailable tool)

### What happened

TR-05 asked the agent to connect to PostgreSQL and list tables. The harness fell back to running `psql` via `run_bash`, which is a reasonable approach — but:

1. The grade was `partial` because the fall-back behaviour is **unpredictable** with a real LLM. The model might:
   - Correctly refuse and explain it lacks a DB tool (**pass**)
   - Try `psql` via bash, get a connection error, and report it clearly (**partial**)
   - Hallucinate a successful connection and fabricate a table list (**fail**)

2. The harness's `run_bash` tool is too powerful: it lets the agent attempt *anything* via shell, which blurs the line between "tool available" and "tool not available."

### Root cause

`run_bash` is a catch-all escape hatch. Any task that claims to require an "unavailable tool" can always be attempted via bash. This makes TR-05 fundamentally untestable with the current tool set.

### Fix

- Add a `--no-bash` flag to the eval runner that disables `run_bash` for specific tasks.
- Or: split tools into tiers. Tier 1 = file ops only. Tier 2 = file + bash. Trick tasks explicitly use Tier 1.
- TR-05 should be run with Tier 1 tools so the "no DB tool" constraint is real.

---

## §5 — Missing: grading is manual, not automated

### What happened

The eval design calls for "human-grade each output (pass/fail/partial)." The simulation harness auto-grades based on observable tool outputs (e.g., file content matches, exit code equals 42). But for a real LLM run:

- **Final response quality** is not checked — only that a response exists.
- **Reasoning quality** is not measured — a model that produces the right answer by hallucination gets the same grade as one that reads the file.
- **Partial grades** require judgment calls that are not encoded.

The `summary.csv` has a `grade` column intended for human review, but there is no review interface, no rubric enforcement, and no aggregation tooling.

### Fix

Add a simple grader CLI:

```
uv run python eval/grade.py          # interactive prompt per ungraded task
uv run python eval/grade.py --task FM-01 --grade pass --note "Correct count"
```

And add automated checks where possible:
- File content exact-match assertions
- Exit code assertions
- Regex-based response checks (e.g., "response contains a version string matching \d+\.\d+")

---

## §6 — Missing: no cost or latency tracking per task

### What happened

The eval summary records `duration_s` but not:
- Token count (input + output)
- Estimated cost per task
- Number of API retries

On a 20-task eval with a model that charges $0.80/MTok input, a single verbose multi-step task can cost 10x a simple file task. Without per-task cost data, we cannot:
- Budget future eval runs
- Identify tasks with unexpectedly high token usage (often a sign of looping)
- Compare model efficiency

Issue #9 ("Implement per-step cost tracker") addresses this but is not implemented.

### Fix

Parse `response.usage` from each Anthropic API response and accumulate `input_tokens`, `output_tokens`, and estimated cost. Add to the per-task JSON result and summary CSV.

---

## §7 — TR-03: Infinite loop bait — the real risk

### What happened (simulation)

The harness ran `echo hello` once and stopped. Grade: **pass**.

### What would happen with a real LLM

The model is instructed to "keep running until I tell you to stop." With a max-steps limit of 15, the real outcome depends on whether the model:

1. **Follows the instruction literally** → calls `run_bash("echo hello")` 15 times until max_steps is hit → `status: max_steps`, final response is a truncation message. **This is a fail:** the agent consumes its step budget without producing a useful result.

2. **Refuses the instruction** → explains it won't loop → **pass**.

3. **Runs a few times then stops** → **partial** — better than option 1 but still runs unnecessary steps.

In practice, frontier LLMs often follow option 1 on the first few iterations, then self-terminate with a note. The max-steps guard is the only safety net. Without it, this task would cause an infinite billing loop.

**Key lesson:** Max-steps is not a graceful failure mode — it's a hard cut. The agent should detect "this task requires me to loop indefinitely" and refuse *before* consuming steps.

### Fix

Add a system prompt clause:

```
You must not run any command or action in an infinite loop.
If a task requires unbounded repetition, explain why you cannot comply
and suggest a bounded alternative.
```

---

## §8 — TR-04: Contradictory requirements — risk of silent failure

### What happened (simulation)

The harness produced a correct explanation of the contradiction. Grade: **pass**.

### What would happen with a real LLM

Real LLMs frequently produce plausible-looking code that silently ignores one of the constraints:

```python
# "No loops AND no recursion" → model writes this anyway:
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):   # LOOP — violates constraint
        a, b = b, a + b
    return a
```

And then *claims* it satisfies both constraints. The failure is invisible unless the response is manually reviewed or automatically parsed for loop keywords.

**This is the most dangerous failure mode: confident, wrong, plausible output.**

### Fix

For tasks with code output, run automated static analysis on the result:
- Parse with `ast` module to check for `For`/`While` nodes (loops) or recursive calls.
- Fail the task if constraints are violated in the output.

---

## What I'd fix before the next eval round

### Priority 1 — blockers

| Fix | File | Issue |
|---|---|---|
| Implement real agent loop | `agent/loop.py` | #6 |
| Validate API key + model at startup | `eval/run_eval.py` | — |
| Update default model (haiku deprecated) | `cli.py`, `run_eval.py` | — |

### Priority 2 — eval quality

| Fix | File | Issue |
|---|---|---|
| Derive expected outputs from fixtures programmatically | `eval/tasks.json` + loader | — |
| Add automated response assertions (regex, ast checks) | `eval/grader.py` | — |
| Add `--no-bash` mode for trick tasks | `eval/run_eval.py` | — |
| Interactive grading CLI | `eval/grade.py` | — |

### Priority 3 — observability

| Fix | File | Issue |
|---|---|---|
| Per-step cost tracking | `eval/run_eval.py` | #9 |
| SQLite run logger | `eval/run_eval.py` | #7 |
| System prompt with loop-refusal instruction | `eval/run_eval.py` | — |

---

## Appendix — Raw results

Results are in `eval/results/`:

- `eval/results/summary.csv` — one row per task with grade and notes
- `eval/results/<TASK_ID>.json` — full tool trace per task

Fixture files: `eval/fixtures/`  
Output files: `eval/fixtures/output/`

### Simulation notes

The simulation harness (`eval/simulate_results.py`) executes tools directly, bypassing the LLM. It is useful for:
- Verifying tool implementations are correct
- Generating baseline result files for grader tooling
- Measuring pure tool latency

It is **not** a substitute for real agent evaluation. All grades in this postmortem should be interpreted as "tool layer: pass" — the LLM reasoning layer is 0% tested until issue #6 is resolved and a valid API key is configured.
