# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A **teaching skeleton**, not a finished system. It's the starter repo for "Lab 20: Multi-Agent Research System" — a Supervisor + Researcher + Analyst + Writer pipeline benchmarked against a single-agent baseline. The core logic is intentionally left as `StudentTodoError`-raising stubs so learners implement it themselves.

**Key implication:** Many methods raising `StudentTodoError` are *correct as-is* — they are the assignment, not bugs. Tests assert these errors are raised (see `tests/test_agents_todo.py`). Do not "fix" a stub unless explicitly asked to implement that part; doing so breaks the lab's intended state and its tests.

## Commands

```bash
make install      # pip install -e "[dev,llm]"
make test         # pytest (config in pyproject.toml: pythonpath=src, testpaths=tests)
make lint         # ruff check src tests
make format       # ruff format src tests
make typecheck    # mypy src (strict mode)

pytest tests/test_state.py                       # single test file
pytest tests/test_state.py::test_name -q         # single test
bash scripts/check_todos.sh                      # list all TODO(student) markers

# Run the CLI (entrypoint: `malab` or `python -m multi_agent_research_lab.cli`)
python -m multi_agent_research_lab.cli baseline    --query "..."   # minimal single-agent placeholder
python -m multi_agent_research_lab.cli multi-agent --query "..."   # exits code 2 until workflow is implemented
```

mypy runs in **strict** mode and ruff enforces `E,F,I,B,UP,SIM` at line-length 100 — new code must satisfy both.

## Architecture

State-passing pipeline. A single mutable `ResearchState` (Pydantic) is the source of truth threaded through every agent; agents read fields they need and write fields they produce, then return the same state.

- **`core/`** — the contracts. `schemas.py` (Pydantic I/O models + `AgentName` enum), `state.py` (`ResearchState`, with `record_route` / `add_trace_event` helpers), `config.py` (`Settings` via pydantic-settings + cached `get_settings()`), `errors.py` (`StudentTodoError`, `AgentExecutionError`, `ValidationError`).
- **`agents/`** — all subclass `BaseAgent` (ABC with `run(state) -> state`). `SupervisorAgent` is the router that decides the next worker and stop condition; `researcher`/`analyst`/`writer`/`critic` are workers. All currently stubbed.
- **`graph/workflow.py`** — `MultiAgentWorkflow` orchestrates agents into a graph (intended to use LangGraph). Orchestration lives here; agent internals stay in `agents/`.
- **`services/`** — provider-agnostic clients (`llm_client.py`, `search_client.py`, `storage.py`). **Agents depend on these interfaces, never import an SDK directly.** Retry/timeout/token-logging belong in the service layer, not in agents.
- **`evaluation/`** — `benchmark.py` (`run_benchmark` measures latency, returns `BenchmarkMetrics`), `report.py`.
- **`observability/`** — `logging.py`, `tracing.py` hooks.
- **`cli.py`** — Typer app wiring config → state → workflow.

Data flow: `cli` builds `ResearchQuery` → `ResearchState` → `SupervisorAgent` routes among workers (bounded by `Settings.max_iterations` / `timeout_seconds`) → `WriterAgent` produces `final_answer` → `evaluation` compares single vs multi-agent.

## Conventions to preserve

- All cross-boundary input/output is a Pydantic schema (`core/schemas.py`); extend `ResearchState` when adding agent outputs or metrics.
- Never read environment variables in agents — go through `get_settings()`.
- Guardrails are required by design: respect `max_iterations` and `timeout_seconds`; agents should fail via `AgentExecutionError` after retries rather than loop forever.
- Per user's global rules: explain changes briefly, don't add `console.log`/prints (use the logging hooks), run typecheck after edits, and never auto-commit or delete files without asking.
