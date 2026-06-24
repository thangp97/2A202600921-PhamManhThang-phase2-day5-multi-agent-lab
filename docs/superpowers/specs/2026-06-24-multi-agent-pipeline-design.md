# Design: Multi-Agent Research Pipeline Implementation

- **Date:** 2026-06-24
- **Status:** Approved
- **Scope:** Implement the full multi-agent research pipeline on top of the existing teaching skeleton, replacing all `StudentTodoError` stubs with working code.

## 1. Summary & guiding principles

Turn the `multi-agent-research-lab` skeleton into a working system: Supervisor + Researcher + Analyst + Writer + Critic, orchestrated with LangGraph, benchmarked against a single-agent baseline.

Guiding philosophy: **offline-first hybrid**. Everything runs and tests green offline using deterministic *fake* backends; real providers (OpenRouter LLM, Tavily search, LangSmith tracing) activate only when their API keys are present.

Locked stack decisions:

- **LLM:** OpenRouter via the `openai` SDK (`base_url=https://openrouter.ai/api/v1`, model `openai/gpt-4o-mini`) + a fake backend.
- **Search:** fake backend by default + Tavily optional.
- **Orchestration:** LangGraph `StateGraph`, graph state = `ResearchState` (Pydantic).
- **Agents:** 5 roles — Supervisor + Researcher + Analyst + Writer + Critic (with a bounded revision loop).
- **Observability:** in-state trace always on + LangSmith optional.
- **Benchmark:** heuristic score always computed + LLM-judge optional; cost derived from token usage.

## 2. Backend abstraction layer (hybrid)

Each external service has an interface + a factory that selects the backend from config.

### `services/llm_client.py`
- `LLMClient` (ABC): `complete(system_prompt, user_prompt) -> LLMResponse`.
- `FakeLLMClient`: returns deterministic, role-aware template content; populates fake `input_tokens`/`output_tokens` so downstream cost/metrics code has data.
- `OpenRouterLLMClient`: calls OpenRouter through the `openai` SDK; fills `input_tokens`, `output_tokens`, `cost_usd`; retry + timeout via `tenacity`.
- `get_llm_client(settings) -> LLMClient`: factory keyed on `settings.llm_backend`.

### `services/search_client.py`
- `SearchClient` (ABC): `search(query, max_results) -> list[SourceDocument]`.
- `FakeSearchClient`: returns a fixed list of `SourceDocument`.
- `TavilySearchClient`: real web search via Tavily.
- `get_search_client(settings) -> SearchClient`: factory keyed on `settings.search_backend`.

### `services/storage.py`
- Implement writing reports/traces to `reports/` as JSON + Markdown.

**Rule:** retry/timeout/token-logging live in the service layer, never inside agents. Agents receive clients via constructor injection.

## 3. Agents & orchestration

### Agent contract
Each agent receives `llm_client` (Researcher also receives `search_client`) via constructor, implements `run(state) -> state`, appends an `AgentResult` to `state.agent_results`, and records an event via `state.add_trace_event`.

- **ResearcherAgent:** `search_client.search()` → populate `state.sources`; LLM summarizes into `state.research_notes`.
- **AnalystAgent:** reads `research_notes` + `sources` → LLM produces `state.analysis_notes` (insights, gaps).
- **WriterAgent:** reads notes → LLM writes `state.final_answer` for the target `audience` and length, with citations.
- **CriticAgent:** scores `final_answer` via an LLM rubric → returns an `AgentResult` carrying `quality_score`, a verdict (`accept|revise`), and feedback (in `metadata`).
- **SupervisorAgent (router):** picks the next route from which fields are populated, the Critic verdict, and `iteration`.

### Supervisor routing policy
- no `research_notes` → `researcher`
- has research, no `analysis_notes` → `analyst`
- has analysis, no/needs-revision `final_answer` → `writer`
- just wrote → `critic`
- critic `accept` **or** `iteration >= max_iterations` → `done`
- critic `revise` → back to `writer` (or `researcher` if feedback demands more sources)

### LangGraph topology
- `StateGraph(ResearchState)`.
- `supervisor` node is the dispatch point; **conditional edges** map the route-string → worker node.
- Each worker returns to `supervisor`; route `done` → `END`.
- `record_route`/`iteration` increments each loop; combined with `max_iterations` this guarantees termination.
- `MultiAgentWorkflow.build()` constructs the graph; `run()` compiles, invokes, and returns a `ResearchState`.

### Single-agent baseline
`cli.py baseline` uses the same `LLMClient` but a single prompt that does research + analysis + writing in one shot, measured identically for a fair comparison.

## 4. Evaluation, CLI & config

### Evaluation
- `evaluation/benchmark.py`: `run_benchmark(run_name, query, runner)` keeps latency measurement; adds aggregated `estimated_cost_usd` (sum of token-derived `cost_usd`) and `quality_score`.
- `evaluation/scoring.py` (new):
  - `heuristic_score(state)` — length-target met, citation coverage (# sources referenced), structural completeness. Always runs, fully deterministic.
  - `llm_judge_score(state, llm_client)` — rubric 0–10 + rationale. Runs when backend is real.
  - `BenchmarkMetrics` carries both.
- `evaluation/report.py`: renders a single-vs-multi comparison table to `reports/benchmark_report.md` (deliverable).

### CLI (`cli.py`)
- `baseline --query` → real single-agent + print metrics.
- `multi-agent --query [--backend fake|openrouter]` → run workflow, print `final_answer` + trace summary.
- `benchmark --query [--backend ...]` (new) → run both, write `reports/benchmark_report.md` + dump trace JSON.
- `--backend` overrides `Settings`; default `fake` so it runs with no keys.

### Config (`core/config.py`)
Add: `llm_backend` (`fake|openrouter`, default `fake`), `search_backend` (`fake|tavily`, default `fake`), `openrouter_api_key`, `openrouter_base_url` (default `https://openrouter.ai/api/v1`), `openrouter_model` (default `openai/gpt-4o-mini`).
**Replace** the existing OpenAI LLM fields with OpenRouter (still using the `openai` SDK). Update `.env.example` accordingly.

## 5. Testing, dependencies & error handling

### Testing
- **Replace** `tests/test_agents_todo.py` (which asserts `StudentTodoError`) with real behavior tests using `FakeLLMClient`/`FakeSearchClient`.
- New tests:
  - `test_supervisor_routing.py` — routing policy across states, `max_iterations`, critic verdict.
  - `test_workflow.py` — full graph with fake backend: produces `final_answer`, terminates correctly, no infinite loop.
  - `test_scoring.py` — heuristic scoring is deterministic.
  - `test_llm_client_factory.py` / `test_search_client_factory.py` — factory selects the right backend; fake needs no key.
- Keep still-valid tests: `test_config.py`, `test_state.py`, `test_report.py` (update if schemas change).
- Acceptance: `make test` green **fully offline** with fake backend; mypy strict + ruff clean.

### Dependencies
- Move `langgraph`, `langchain-core`, `langsmith` from optional `[llm]` into core deps (orchestration now depends on LangGraph).
- Keep `openai` and `tavily-python` optional for the real backend (fake needs neither).

### Error handling
- Real clients: retry/timeout via `tenacity`.
- Failure after retries → `AgentExecutionError`; Supervisor catches it and routes `done` with a fallback message instead of crashing.
- I/O validated via Pydantic; schema failures → `ValidationError`.
- `StudentTodoError` remains only at genuine extension points, or is removed entirely.

## 6. Out of scope (YAGNI)
- No persistent vector store / RAG index.
- No web UI; CLI only.
- No multi-provider LLM beyond OpenRouter + fake.
- No checkpointing/resume beyond what LangGraph gives for free.
