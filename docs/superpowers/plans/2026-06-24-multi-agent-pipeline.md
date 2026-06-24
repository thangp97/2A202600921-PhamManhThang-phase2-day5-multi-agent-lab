# Multi-Agent Research Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all `StudentTodoError` stubs in the lab skeleton with a working Supervisor + Researcher + Analyst + Writer + Critic pipeline orchestrated by LangGraph, benchmarked against a single-agent baseline.

**Architecture:** Offline-first hybrid. Service interfaces (`LLMClient`, `SearchClient`) each have a deterministic *fake* backend (default) and a *real* backend (OpenRouter LLM, Tavily search) selected by config. Agents receive clients via constructor injection and mutate a single Pydantic `ResearchState`. LangGraph `StateGraph` wires a supervisor router to worker nodes with a bounded critic-revision loop. Evaluation runs heuristic scoring always and LLM-judge scoring when a real backend is active.

**Tech Stack:** Python 3.11, Pydantic v2, pydantic-settings, Typer, Rich, LangGraph, `openai` SDK (pointed at OpenRouter), Tavily, tenacity, pytest, ruff, mypy (strict).

## Global Constraints

- Python `>=3.11`; ruff lint set `E,F,I,B,UP,SIM` at line-length 100; mypy `strict = true` — all new code must pass `make lint` and `make typecheck`.
- `make test` MUST be green **fully offline** with the default `fake` backend (no API keys, no network).
- Agents NEVER read environment variables or import provider SDKs directly — they depend on `LLMClient` / `SearchClient` interfaces. Config is read only via `get_settings()`.
- Retry/timeout/token-logging live in the service layer, not in agents.
- All cross-boundary I/O uses Pydantic schemas from `core/schemas.py`.
- Default LLM model is `openai/gpt-4o-mini` via OpenRouter base URL `https://openrouter.ai/api/v1`.
- **Project commit rule (user CLAUDE.md):** do not auto-commit. When a step says "Commit", confirm with the user first (or batch per the executor's checkpoint).
- Reply language to the user is Vietnamese; code identifiers in English, comments may be Vietnamese.

---

## File Structure

**Create:**
- `src/multi_agent_research_lab/agents/researcher.py` — already exists as stub; rewrite.
- `src/multi_agent_research_lab/agents/analyst.py`, `writer.py`, `critic.py`, `supervisor.py` — exist as stubs; rewrite.
- `src/multi_agent_research_lab/evaluation/scoring.py` — new.
- `src/multi_agent_research_lab/agents/prompts.py` — new, holds prompt templates.
- Tests: `tests/test_llm_client.py`, `tests/test_search_client.py`, `tests/test_researcher.py`, `tests/test_analyst.py`, `tests/test_writer.py`, `tests/test_critic.py`, `tests/test_supervisor_routing.py`, `tests/test_workflow.py`, `tests/test_scoring.py`, `tests/test_benchmark.py`.

**Modify:**
- `core/config.py` — OpenRouter + backend fields.
- `core/schemas.py` — `Verdict` enum, `Route` constants, fields on `AgentResult`.
- `core/state.py` — `next_route` field.
- `services/llm_client.py`, `services/search_client.py`, `services/storage.py`.
- `graph/workflow.py` — LangGraph build/run.
- `evaluation/benchmark.py`, `evaluation/report.py`.
- `cli.py` — `--backend`, `benchmark` command, real baseline.
- `pyproject.toml` — move LangGraph stack to core deps.
- `.env.example` — OpenRouter vars.
- **Delete:** `tests/test_agents_todo.py` (replaced by behavior tests).

---

## Task 1: Config & dependencies for OpenRouter + backends

**Files:**
- Modify: `src/multi_agent_research_lab/core/config.py`
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Test: `tests/test_config.py` (extend existing)

**Interfaces:**
- Produces: `Settings` with fields `llm_backend: str` (`"fake"|"openrouter"`, default `"fake"`), `search_backend: str` (`"fake"|"tavily"`, default `"fake"`), `openrouter_api_key: str | None`, `openrouter_base_url: str` (default `"https://openrouter.ai/api/v1"`), `openrouter_model: str` (default `"openai/gpt-4o-mini"`), `tavily_api_key: str | None`, plus existing `max_iterations`, `timeout_seconds`, `log_level`. `get_settings() -> Settings`.

- [ ] **Step 1: Write the failing test**

In `tests/test_config.py` add:

```python
def test_settings_defaults_to_fake_backends(monkeypatch):
    from multi_agent_research_lab.core.config import Settings
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("SEARCH_BACKEND", raising=False)
    s = Settings(_env_file=None)
    assert s.llm_backend == "fake"
    assert s.search_backend == "fake"
    assert s.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert s.openrouter_model == "openai/gpt-4o-mini"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_settings_defaults_to_fake_backends -v`
Expected: FAIL (`AttributeError`/`ValidationError`: no `llm_backend`).

- [ ] **Step 3: Implement config changes**

Replace the OpenAI block in `core/config.py` with:

```python
    llm_backend: str = Field(default="fake", validation_alias="LLM_BACKEND")
    search_backend: str = Field(default="fake", validation_alias="SEARCH_BACKEND")

    openrouter_api_key: str | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", validation_alias="OPENROUTER_BASE_URL"
    )
    openrouter_model: str = Field(
        default="openai/gpt-4o-mini", validation_alias="OPENROUTER_MODEL"
    )

    tavily_api_key: str | None = Field(default=None, validation_alias="TAVILY_API_KEY")

    langsmith_api_key: str | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(
        default="multi-agent-research-lab", validation_alias="LANGSMITH_PROJECT"
    )
```

(Remove the `openai_api_key` / `openai_model` fields.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Update pyproject.toml dependencies**

In `[project].dependencies` add `"langgraph>=0.2"`, `"langchain-core>=0.2"`, `"langsmith>=0.1"`. In `[project.optional-dependencies]` set:

```toml
llm = [
  "openai>=1.40",
  "tavily-python>=0.5",
]
```

- [ ] **Step 6: Update .env.example**

Replace OpenAI lines with:

```bash
LLM_BACKEND=fake
SEARCH_BACKEND=fake
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=openai/gpt-4o-mini
TAVILY_API_KEY=
LANGSMITH_API_KEY=
```

- [ ] **Step 7: Verify lint/type/test**

Run: `make lint && make typecheck && pytest tests/test_config.py -v`
Expected: all PASS.

- [ ] **Step 8: Commit** (confirm with user first)

```bash
git add src/multi_agent_research_lab/core/config.py pyproject.toml .env.example tests/test_config.py
git commit -m "feat(config): add OpenRouter + backend selection settings"
```

---

## Task 2: Schema & state additions

**Files:**
- Modify: `src/multi_agent_research_lab/core/schemas.py`
- Modify: `src/multi_agent_research_lab/core/state.py`
- Test: `tests/test_state.py` (extend existing)

**Interfaces:**
- Produces:
  - `Verdict` (StrEnum): `ACCEPT = "accept"`, `REVISE = "revise"`.
  - `Route` (StrEnum): `RESEARCHER="researcher"`, `ANALYST="analyst"`, `WRITER="writer"`, `CRITIC="critic"`, `DONE="done"`.
  - `AgentResult` gains `quality_score: float | None = None`.
  - `ResearchState` gains `next_route: str | None = None`.

- [ ] **Step 1: Write the failing test**

In `tests/test_state.py` add:

```python
def test_state_has_next_route_and_route_enum():
    from multi_agent_research_lab.core.schemas import Route, ResearchQuery
    from multi_agent_research_lab.core.state import ResearchState
    state = ResearchState(request=ResearchQuery(query="hello world"))
    assert state.next_route is None
    state.next_route = Route.RESEARCHER
    assert state.next_route == "researcher"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_state.py::test_state_has_next_route_and_route_enum -v`
Expected: FAIL (`ImportError: Route` or unknown field).

- [ ] **Step 3: Implement schema additions**

In `core/schemas.py` add near `AgentName`:

```python
class Verdict(StrEnum):
    ACCEPT = "accept"
    REVISE = "revise"


class Route(StrEnum):
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    CRITIC = "critic"
    DONE = "done"
```

Add to `AgentResult`:

```python
    quality_score: float | None = Field(default=None, ge=0, le=10)
```

- [ ] **Step 4: Implement state addition**

In `core/state.py`, add to `ResearchState` (after `route_history`):

```python
    next_route: str | None = None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_state.py -v`
Expected: PASS.

- [ ] **Step 6: Commit** (confirm with user first)

```bash
git add src/multi_agent_research_lab/core/schemas.py src/multi_agent_research_lab/core/state.py tests/test_state.py
git commit -m "feat(core): add Route/Verdict enums and next_route state field"
```

---

## Task 3: LLM client (fake + OpenRouter + factory)

**Files:**
- Modify: `src/multi_agent_research_lab/services/llm_client.py`
- Test: `tests/test_llm_client.py` (create)

**Interfaces:**
- Consumes: `Settings` (Task 1).
- Produces:
  - `LLMResponse` (existing dataclass).
  - `LLMClient` (ABC): `complete(system_prompt: str, user_prompt: str) -> LLMResponse`.
  - `FakeLLMClient(LLMClient)`: deterministic; `complete` returns `LLMResponse(content=f"[fake-llm] sys={system_prompt[:32]} | user={user_prompt[:64]}", input_tokens=10, output_tokens=20, cost_usd=0.0)`.
  - `OpenRouterLLMClient(LLMClient)`: ctor `(api_key, base_url, model, timeout)`; uses `openai` SDK; retry via tenacity.
  - `get_llm_client(settings: Settings) -> LLMClient`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_llm_client.py`:

```python
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.services.llm_client import (
    FakeLLMClient,
    LLMResponse,
    get_llm_client,
)


def test_fake_llm_is_deterministic():
    client = FakeLLMClient()
    r1 = client.complete("sysA", "userA")
    r2 = client.complete("sysA", "userA")
    assert isinstance(r1, LLMResponse)
    assert r1.content == r2.content
    assert r1.input_tokens == 10 and r1.output_tokens == 20


def test_factory_returns_fake_by_default():
    settings = Settings(_env_file=None)
    assert isinstance(get_llm_client(settings), FakeLLMClient)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_client.py -v`
Expected: FAIL (`ImportError: FakeLLMClient`).

- [ ] **Step 3: Implement the client**

Replace `services/llm_client.py` with:

```python
"""LLM client abstraction.

Agents depend on this interface instead of importing an SDK directly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import AgentExecutionError


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion."""


class FakeLLMClient(LLMClient):
    """Deterministic offline client for tests and no-key runs."""

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        content = f"[fake-llm] sys={system_prompt[:32]} | user={user_prompt[:64]}"
        return LLMResponse(content=content, input_tokens=10, output_tokens=20, cost_usd=0.0)


class OpenRouterLLMClient(LLMClient):
    """Real backend via the OpenAI SDK pointed at OpenRouter."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: int) -> None:
        from openai import OpenAI  # lazy import: optional dependency

        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self._model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - normalize to domain error
            raise AgentExecutionError(f"OpenRouter call failed: {exc}") from exc
        usage = resp.usage
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            cost_usd=None,
        )


def get_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_backend == "openrouter":
        if not settings.openrouter_api_key:
            raise AgentExecutionError("OPENROUTER_API_KEY required for openrouter backend")
        return OpenRouterLLMClient(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            model=settings.openrouter_model,
            timeout=settings.timeout_seconds,
        )
    return FakeLLMClient()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_client.py -v`
Expected: PASS.

- [ ] **Step 5: Verify lint/type**

Run: `make lint && make typecheck`
Expected: PASS.

- [ ] **Step 6: Commit** (confirm with user first)

```bash
git add src/multi_agent_research_lab/services/llm_client.py tests/test_llm_client.py
git commit -m "feat(services): implement fake + OpenRouter LLM clients with factory"
```

---

## Task 4: Search client (fake + Tavily + factory)

**Files:**
- Modify: `src/multi_agent_research_lab/services/search_client.py`
- Test: `tests/test_search_client.py` (create)

**Interfaces:**
- Consumes: `Settings` (Task 1), `SourceDocument` (schemas).
- Produces:
  - `SearchClient` (ABC): `search(query: str, max_results: int) -> list[SourceDocument]`.
  - `FakeSearchClient`: returns `max_results` deterministic `SourceDocument`s titled `f"Fake source {i} for {query}"`.
  - `TavilySearchClient(api_key)`: real search (lazy import).
  - `get_search_client(settings) -> SearchClient`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_search_client.py`:

```python
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.services.search_client import (
    FakeSearchClient,
    get_search_client,
)


def test_fake_search_returns_n_sources():
    client = FakeSearchClient()
    docs = client.search("graphrag", max_results=3)
    assert len(docs) == 3
    assert all(isinstance(d, SourceDocument) for d in docs)
    assert docs[0].title == "Fake source 0 for graphrag"


def test_factory_returns_fake_by_default():
    assert isinstance(get_search_client(Settings(_env_file=None)), FakeSearchClient)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_search_client.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement the client**

Replace `services/search_client.py` with:

```python
"""Search client abstraction (provider-agnostic)."""

from abc import ABC, abstractmethod

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Return source documents for a query."""


class FakeSearchClient(SearchClient):
    def search(self, query: str, max_results: int) -> list[SourceDocument]:
        return [
            SourceDocument(
                title=f"Fake source {i} for {query}",
                url=f"https://example.test/{i}",
                snippet=f"Deterministic snippet {i} about {query}.",
            )
            for i in range(max_results)
        ]


class TavilySearchClient(SearchClient):
    def __init__(self, api_key: str) -> None:
        from tavily import TavilyClient  # lazy import: optional dependency

        self._client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int) -> list[SourceDocument]:
        try:
            res = self._client.search(query=query, max_results=max_results)
        except Exception as exc:  # noqa: BLE001
            raise AgentExecutionError(f"Tavily search failed: {exc}") from exc
        return [
            SourceDocument(
                title=item.get("title", "Untitled"),
                url=item.get("url"),
                snippet=item.get("content", ""),
            )
            for item in res.get("results", [])
        ]


def get_search_client(settings: Settings) -> SearchClient:
    if settings.search_backend == "tavily":
        if not settings.tavily_api_key:
            raise AgentExecutionError("TAVILY_API_KEY required for tavily backend")
        return TavilySearchClient(api_key=settings.tavily_api_key)
    return FakeSearchClient()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_search_client.py -v`
Expected: PASS.

- [ ] **Step 5: Commit** (confirm with user first)

```bash
git add src/multi_agent_research_lab/services/search_client.py tests/test_search_client.py
git commit -m "feat(services): implement fake + Tavily search clients with factory"
```

---

## Task 5: Prompt templates

**Files:**
- Create: `src/multi_agent_research_lab/agents/prompts.py`

**Interfaces:**
- Produces module-level constants: `RESEARCHER_SYSTEM`, `ANALYST_SYSTEM`, `WRITER_SYSTEM`, `CRITIC_SYSTEM`, `JUDGE_SYSTEM`, `BASELINE_SYSTEM` (all `str`), and `critic_user(final_answer: str) -> str`, `judge_user(query: str, answer: str) -> str` helpers.

- [ ] **Step 1: Create the file** (no test needed — pure constants, exercised by agent tests)

```python
"""Prompt templates for agents. Keep wording here, logic in agents."""

RESEARCHER_SYSTEM = (
    "You are a research agent. Given a query and sources, write concise, factual "
    "research notes. Cite source titles inline."
)
ANALYST_SYSTEM = (
    "You are an analyst. Given research notes, extract key insights, tensions, and gaps. "
    "Be structured and brief."
)
WRITER_SYSTEM = (
    "You are a technical writer. Using the research and analysis notes, write a clear "
    "answer for the given audience, with inline citations to source titles."
)
CRITIC_SYSTEM = (
    "You are a critic. Score the answer 0-10 for completeness, accuracy, citations, and "
    "clarity. Reply with the score on the first line as 'SCORE: <n>' then 'VERDICT: "
    "accept' or 'VERDICT: revise' then feedback."
)
JUDGE_SYSTEM = (
    "You are an impartial judge. Score the answer to the query from 0 to 10 for overall "
    "quality. Reply with 'SCORE: <n>' on the first line, then a one-line rationale."
)
BASELINE_SYSTEM = (
    "You are a single research assistant. Research, analyze, and write a complete answer "
    "to the query in one response, with citations where possible."
)


def critic_user(final_answer: str) -> str:
    return f"Evaluate this answer:\n\n{final_answer}"


def judge_user(query: str, answer: str) -> str:
    return f"Query: {query}\n\nAnswer:\n{answer}"
```

- [ ] **Step 2: Commit** (confirm with user first)

```bash
git add src/multi_agent_research_lab/agents/prompts.py
git commit -m "feat(agents): add prompt templates module"
```

---

## Task 6: Researcher agent

**Files:**
- Modify: `src/multi_agent_research_lab/agents/researcher.py`
- Test: `tests/test_researcher.py` (create)

**Interfaces:**
- Consumes: `LLMClient`, `SearchClient`, `ResearchState`, `RESEARCHER_SYSTEM`.
- Produces: `ResearcherAgent(llm_client: LLMClient, search_client: SearchClient)`; `run(state)` sets `state.sources`, `state.research_notes`, appends `AgentResult(agent=AgentName.RESEARCHER, ...)`, records trace; returns state.

- [ ] **Step 1: Write the failing test**

Create `tests/test_researcher.py`:

```python
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import FakeLLMClient
from multi_agent_research_lab.services.search_client import FakeSearchClient


def test_researcher_populates_sources_and_notes():
    state = ResearchState(request=ResearchQuery(query="explain graphrag", max_sources=2))
    agent = ResearcherAgent(FakeLLMClient(), FakeSearchClient())
    out = agent.run(state)
    assert len(out.sources) == 2
    assert out.research_notes is not None and out.research_notes != ""
    assert out.agent_results[-1].agent == AgentName.RESEARCHER
    assert any(e["name"] == "researcher" for e in out.trace)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_researcher.py -v`
Expected: FAIL (`StudentTodoError`).

- [ ] **Step 3: Implement the agent**

Replace `agents/researcher.py` with:

```python
"""Researcher agent: gathers sources and writes research notes."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.prompts import RESEARCHER_SYSTEM
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    name = "researcher"

    def __init__(self, llm_client: LLMClient, search_client: SearchClient) -> None:
        self._llm = llm_client
        self._search = search_client

    def run(self, state: ResearchState) -> ResearchState:
        state.sources = self._search.search(
            state.request.query, max_results=state.request.max_sources
        )
        sources_text = "\n".join(f"- {s.title}: {s.snippet}" for s in state.sources)
        user = f"Query: {state.request.query}\n\nSources:\n{sources_text}"
        resp = self._llm.complete(RESEARCHER_SYSTEM, user)
        state.research_notes = resp.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=resp.content,
                metadata={
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cost_usd": resp.cost_usd,
                    "num_sources": len(state.sources),
                },
            )
        )
        state.add_trace_event("researcher", {"num_sources": len(state.sources)})
        return state
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_researcher.py -v`
Expected: PASS.

- [ ] **Step 5: Commit** (confirm with user first)

```bash
git add src/multi_agent_research_lab/agents/researcher.py tests/test_researcher.py
git commit -m "feat(agents): implement researcher agent"
```

---

## Task 7: Analyst agent

**Files:**
- Modify: `src/multi_agent_research_lab/agents/analyst.py`
- Test: `tests/test_analyst.py` (create)

**Interfaces:**
- Consumes: `LLMClient`, `ResearchState`, `ANALYST_SYSTEM`.
- Produces: `AnalystAgent(llm_client: LLMClient)`; `run(state)` sets `state.analysis_notes`, appends `AgentResult(agent=AgentName.ANALYST)`, records trace.

- [ ] **Step 1: Write the failing test**

Create `tests/test_analyst.py`:

```python
from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import FakeLLMClient


def test_analyst_produces_analysis_notes():
    state = ResearchState(request=ResearchQuery(query="explain graphrag"))
    state.research_notes = "some research notes"
    out = AnalystAgent(FakeLLMClient()).run(state)
    assert out.analysis_notes is not None and out.analysis_notes != ""
    assert out.agent_results[-1].agent == AgentName.ANALYST
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analyst.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the agent**

Replace `agents/analyst.py`:

```python
"""Analyst agent: turns research notes into structured insights."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.prompts import ANALYST_SYSTEM
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    name = "analyst"

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def run(self, state: ResearchState) -> ResearchState:
        user = f"Research notes:\n{state.research_notes or ''}"
        resp = self._llm.complete(ANALYST_SYSTEM, user)
        state.analysis_notes = resp.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=resp.content,
                metadata={
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cost_usd": resp.cost_usd,
                },
            )
        )
        state.add_trace_event("analyst", {})
        return state
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analyst.py -v`
Expected: PASS.

- [ ] **Step 5: Commit** (confirm with user first)

```bash
git add src/multi_agent_research_lab/agents/analyst.py tests/test_analyst.py
git commit -m "feat(agents): implement analyst agent"
```

---

## Task 8: Writer agent

**Files:**
- Modify: `src/multi_agent_research_lab/agents/writer.py`
- Test: `tests/test_writer.py` (create)

**Interfaces:**
- Consumes: `LLMClient`, `ResearchState`, `WRITER_SYSTEM`.
- Produces: `WriterAgent(llm_client: LLMClient)`; `run(state)` sets `state.final_answer`, appends `AgentResult(agent=AgentName.WRITER)`, records trace.

- [ ] **Step 1: Write the failing test**

Create `tests/test_writer.py`:

```python
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import FakeLLMClient


def test_writer_produces_final_answer():
    state = ResearchState(request=ResearchQuery(query="explain graphrag"))
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    out = WriterAgent(FakeLLMClient()).run(state)
    assert out.final_answer is not None and out.final_answer != ""
    assert out.agent_results[-1].agent == AgentName.WRITER
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_writer.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the agent**

Replace `agents/writer.py`:

```python
"""Writer agent: composes the final answer."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.prompts import WRITER_SYSTEM
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    name = "writer"

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def run(self, state: ResearchState) -> ResearchState:
        user = (
            f"Audience: {state.request.audience}\n"
            f"Query: {state.request.query}\n\n"
            f"Research notes:\n{state.research_notes or ''}\n\n"
            f"Analysis notes:\n{state.analysis_notes or ''}"
        )
        resp = self._llm.complete(WRITER_SYSTEM, user)
        state.final_answer = resp.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=resp.content,
                metadata={
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cost_usd": resp.cost_usd,
                },
            )
        )
        state.add_trace_event("writer", {})
        return state
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_writer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit** (confirm with user first)

```bash
git add src/multi_agent_research_lab/agents/writer.py tests/test_writer.py
git commit -m "feat(agents): implement writer agent"
```

---

## Task 9: Critic agent

**Files:**
- Modify: `src/multi_agent_research_lab/agents/critic.py`
- Test: `tests/test_critic.py` (create)

**Interfaces:**
- Consumes: `LLMClient`, `ResearchState`, `CRITIC_SYSTEM`, `critic_user`, `Verdict`.
- Produces: `CriticAgent(llm_client, accept_threshold: float = 7.0)`; `run(state)` parses score+verdict from LLM, appends `AgentResult(agent=AgentName.CRITIC, quality_score=<float>, metadata={"verdict": <"accept"|"revise">, "feedback": <str>})`, records trace. Exposes `parse_critique(text: str) -> tuple[float, str, str]` returning `(score, verdict, feedback)`; on parse failure defaults to `(accept_threshold, Verdict.ACCEPT, text)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_critic.py`:

```python
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, Verdict
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import FakeLLMClient


def test_parse_critique_reads_score_and_verdict():
    agent = CriticAgent(FakeLLMClient())
    score, verdict, feedback = agent.parse_critique(
        "SCORE: 8\nVERDICT: accept\nLooks complete."
    )
    assert score == 8.0
    assert verdict == Verdict.ACCEPT
    assert "complete" in feedback


def test_parse_critique_defaults_on_garbage():
    agent = CriticAgent(FakeLLMClient(), accept_threshold=7.0)
    score, verdict, _ = agent.parse_critique("no structured output here")
    assert verdict == Verdict.ACCEPT
    assert score == 7.0


def test_critic_run_records_quality_score():
    state = ResearchState(request=ResearchQuery(query="explain graphrag"))
    state.final_answer = "an answer"
    out = CriticAgent(FakeLLMClient()).run(state)
    last = out.agent_results[-1]
    assert last.agent == AgentName.CRITIC
    assert last.quality_score is not None
    assert "verdict" in last.metadata
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_critic.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the agent**

Replace `agents/critic.py`:

```python
"""Critic agent: scores the final answer and emits an accept/revise verdict."""

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.prompts import CRITIC_SYSTEM, critic_user
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, Verdict
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class CriticAgent(BaseAgent):
    name = "critic"

    def __init__(self, llm_client: LLMClient, accept_threshold: float = 7.0) -> None:
        self._llm = llm_client
        self._threshold = accept_threshold

    def parse_critique(self, text: str) -> tuple[float, str, str]:
        score_match = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", text)
        verdict_match = re.search(r"VERDICT:\s*(accept|revise)", text, re.IGNORECASE)
        if score_match is None and verdict_match is None:
            return self._threshold, Verdict.ACCEPT, text
        score = float(score_match.group(1)) if score_match else self._threshold
        score = max(0.0, min(10.0, score))
        if verdict_match:
            verdict = Verdict(verdict_match.group(1).lower())
        else:
            verdict = Verdict.ACCEPT if score >= self._threshold else Verdict.REVISE
        return score, verdict, text

    def run(self, state: ResearchState) -> ResearchState:
        resp = self._llm.complete(CRITIC_SYSTEM, critic_user(state.final_answer or ""))
        score, verdict, feedback = self.parse_critique(resp.content)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=resp.content,
                quality_score=score,
                metadata={"verdict": str(verdict), "feedback": feedback},
            )
        )
        state.add_trace_event("critic", {"score": score, "verdict": str(verdict)})
        return state
```

Note: `FakeLLMClient` output contains no `SCORE:`/`VERDICT:`, so `parse_critique` returns the default `(7.0, accept, ...)` — making fake runs terminate after one writer→critic cycle. This is the intended deterministic offline behavior.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_critic.py -v`
Expected: PASS.

- [ ] **Step 5: Commit** (confirm with user first)

```bash
git add src/multi_agent_research_lab/agents/critic.py tests/test_critic.py
git commit -m "feat(agents): implement critic agent with score/verdict parsing"
```

---

## Task 10: Supervisor routing

**Files:**
- Modify: `src/multi_agent_research_lab/agents/supervisor.py`
- Test: `tests/test_supervisor_routing.py` (create)

**Interfaces:**
- Consumes: `ResearchState`, `Route`, `Verdict`, `Settings`.
- Produces: `SupervisorAgent(max_iterations: int)`; `decide(state) -> Route` (pure routing policy); `run(state)` sets `state.next_route = <Route>`, calls `state.record_route(route)`, records trace; returns state.

Routing policy (in `decide`):
1. `iteration >= max_iterations` → `Route.DONE`
2. last critic result has verdict `accept` → `Route.DONE`
3. no `research_notes` → `Route.RESEARCHER`
4. no `analysis_notes` → `Route.ANALYST`
5. no `final_answer` → `Route.WRITER`
6. has `final_answer` but no critic result yet for the current answer → `Route.CRITIC`
7. last critic verdict `revise` → `Route.WRITER`

- [ ] **Step 1: Write the failing test**

Create `tests/test_supervisor_routing.py`:

```python
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    ResearchQuery,
    Route,
    Verdict,
)
from multi_agent_research_lab.core.state import ResearchState


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="explain graphrag"))


def test_routes_to_researcher_when_empty():
    assert SupervisorAgent(max_iterations=6).decide(_state()) == Route.RESEARCHER


def test_routes_to_analyst_after_research():
    s = _state()
    s.research_notes = "notes"
    assert SupervisorAgent(max_iterations=6).decide(s) == Route.ANALYST


def test_routes_to_writer_after_analysis():
    s = _state()
    s.research_notes, s.analysis_notes = "n", "a"
    assert SupervisorAgent(max_iterations=6).decide(s) == Route.WRITER


def test_routes_to_critic_after_write():
    s = _state()
    s.research_notes, s.analysis_notes, s.final_answer = "n", "a", "ans"
    assert SupervisorAgent(max_iterations=6).decide(s) == Route.CRITIC


def test_done_when_critic_accepts():
    s = _state()
    s.research_notes, s.analysis_notes, s.final_answer = "n", "a", "ans"
    s.agent_results.append(
        AgentResult(agent=AgentName.CRITIC, content="ok", quality_score=9,
                    metadata={"verdict": str(Verdict.ACCEPT)})
    )
    assert SupervisorAgent(max_iterations=6).decide(s) == Route.DONE


def test_revise_routes_back_to_writer():
    s = _state()
    s.research_notes, s.analysis_notes, s.final_answer = "n", "a", "ans"
    s.agent_results.append(
        AgentResult(agent=AgentName.CRITIC, content="meh", quality_score=3,
                    metadata={"verdict": str(Verdict.REVISE)})
    )
    assert SupervisorAgent(max_iterations=6).decide(s) == Route.WRITER


def test_max_iterations_forces_done():
    s = _state()
    s.iteration = 6
    assert SupervisorAgent(max_iterations=6).decide(s) == Route.DONE


def test_run_sets_next_route_and_records():
    s = _state()
    out = SupervisorAgent(max_iterations=6).run(s)
    assert out.next_route == Route.RESEARCHER
    assert out.route_history == [Route.RESEARCHER]
    assert out.iteration == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_supervisor_routing.py -v`
Expected: FAIL (`StudentTodoError` / no `decide`).

- [ ] **Step 3: Implement the supervisor**

Replace `agents/supervisor.py`:

```python
"""Supervisor / router: decides the next agent and the stop condition."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, Route, Verdict
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    name = "supervisor"

    def __init__(self, max_iterations: int) -> None:
        self._max_iterations = max_iterations

    def _last_critic_verdict(self, state: ResearchState) -> str | None:
        for result in reversed(state.agent_results):
            if result.agent == AgentName.CRITIC:
                return result.metadata.get("verdict")
        return None

    def decide(self, state: ResearchState) -> Route:
        if state.iteration >= self._max_iterations:
            return Route.DONE
        verdict = self._last_critic_verdict(state)
        if verdict == str(Verdict.ACCEPT):
            return Route.DONE
        if state.research_notes is None:
            return Route.RESEARCHER
        if state.analysis_notes is None:
            return Route.ANALYST
        if state.final_answer is None:
            return Route.WRITER
        if verdict is None:
            return Route.CRITIC
        # verdict == revise
        return Route.WRITER

    def run(self, state: ResearchState) -> ResearchState:
        route = self.decide(state)
        state.next_route = route
        state.record_route(route)
        state.add_trace_event("supervisor", {"route": str(route)})
        return state
```

Note: when `decide` returns `WRITER` after a `revise`, the previous critic result remains in history; the writer rewrites `final_answer`, and the next supervisor pass sees `verdict == revise` again → routes to `CRITIC`. To avoid an immediate re-critic loop skipping the writer, the writer step is followed by supervisor which now finds `verdict == revise` → returns `WRITER` again. **Fix:** the writer must clear the stale verdict. Add to `WriterAgent.run` (Task 8) — see Step 4 below.

- [ ] **Step 4: Patch WriterAgent to clear stale critic verdict on revision**

The supervisor uses the last critic verdict to decide. After a rewrite, that verdict is stale. Append a sentinel so the next supervisor pass routes writer→critic. In `agents/writer.py`, at the end of `run` before `return state`, add:

```python
        state.add_trace_event("writer_revision_marker", {"answer_version": len(
            [r for r in state.agent_results if r.agent == AgentName.WRITER])})
```

And change `SupervisorAgent._last_critic_verdict` logic to only honor a critic verdict that came **after** the most recent writer result. Replace `_last_critic_verdict` with:

```python
    def _verdict_after_last_write(self, state: ResearchState) -> str | None:
        last_writer_idx = -1
        for i, r in enumerate(state.agent_results):
            if r.agent == AgentName.WRITER:
                last_writer_idx = i
        for r in state.agent_results[last_writer_idx + 1 :]:
            if r.agent == AgentName.CRITIC:
                return r.metadata.get("verdict")
        return None
```

Update `decide` to call `self._verdict_after_last_write(state)` instead of `self._last_critic_verdict(state)`. This makes the loop correct: after a rewrite there is no critic-after-writer yet → routes to `CRITIC`; if that critic says `revise` → routes to `WRITER`; if `accept` → `DONE`.

- [ ] **Step 5: Update the test for the corrected helper**

The `test_done_when_critic_accepts` and `test_revise_routes_back_to_writer` tests append a CRITIC result with no preceding WRITER result in `agent_results`. Update both to append a WRITER result first:

```python
    s.agent_results.append(AgentResult(agent=AgentName.WRITER, content="ans"))
```

(insert before the CRITIC append in those two tests).

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_supervisor_routing.py tests/test_writer.py -v`
Expected: PASS.

- [ ] **Step 7: Commit** (confirm with user first)

```bash
git add src/multi_agent_research_lab/agents/supervisor.py src/multi_agent_research_lab/agents/writer.py tests/test_supervisor_routing.py
git commit -m "feat(agents): implement supervisor routing with bounded critic loop"
```

---

## Task 11: LangGraph workflow

**Files:**
- Modify: `src/multi_agent_research_lab/graph/workflow.py`
- Test: `tests/test_workflow.py` (create)

**Interfaces:**
- Consumes: all agents, `LLMClient`, `SearchClient`, `Settings`, `ResearchState`, `Route`.
- Produces: `MultiAgentWorkflow(llm_client, search_client, max_iterations)`; `build() -> CompiledGraph`; `run(state: ResearchState) -> ResearchState`. Convenience: `build_default_workflow(settings) -> MultiAgentWorkflow`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow.py`:

```python
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import FakeLLMClient
from multi_agent_research_lab.services.search_client import FakeSearchClient


def test_workflow_runs_to_completion_offline():
    wf = MultiAgentWorkflow(FakeLLMClient(), FakeSearchClient(), max_iterations=6)
    state = ResearchState(request=ResearchQuery(query="explain graphrag", max_sources=2))
    out = wf.run(state)
    assert out.final_answer is not None and out.final_answer != ""
    assert out.research_notes is not None
    assert out.analysis_notes is not None
    # fake critic accepts on first pass -> terminates without exhausting iterations
    assert out.iteration <= 6
    assert "done" in [str(r) for r in out.route_history]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow.py -v`
Expected: FAIL (`StudentTodoError`).

- [ ] **Step 3: Implement the workflow**

Replace `graph/workflow.py`:

```python
"""LangGraph workflow wiring the supervisor to worker nodes."""

from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import Route
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, get_llm_client
from multi_agent_research_lab.services.search_client import SearchClient, get_search_client


def _updates(state: ResearchState, *fields: str) -> dict[str, Any]:
    """Return real (non-dumped) model objects for changed fields."""
    return {f: getattr(state, f) for f in fields}


class MultiAgentWorkflow:
    def __init__(
        self, llm_client: LLMClient, search_client: SearchClient, max_iterations: int
    ) -> None:
        self._supervisor = SupervisorAgent(max_iterations=max_iterations)
        self._researcher = ResearcherAgent(llm_client, search_client)
        self._analyst = AnalystAgent(llm_client)
        self._writer = WriterAgent(llm_client)
        self._critic = CriticAgent(llm_client)

    def _supervisor_node(self, state: ResearchState) -> dict[str, Any]:
        self._supervisor.run(state)
        return _updates(state, "next_route", "route_history", "iteration", "trace")

    def _researcher_node(self, state: ResearchState) -> dict[str, Any]:
        self._researcher.run(state)
        return _updates(state, "sources", "research_notes", "agent_results", "trace")

    def _analyst_node(self, state: ResearchState) -> dict[str, Any]:
        self._analyst.run(state)
        return _updates(state, "analysis_notes", "agent_results", "trace")

    def _writer_node(self, state: ResearchState) -> dict[str, Any]:
        self._writer.run(state)
        return _updates(state, "final_answer", "agent_results", "trace")

    def _critic_node(self, state: ResearchState) -> dict[str, Any]:
        self._critic.run(state)
        return _updates(state, "agent_results", "trace")

    @staticmethod
    def _route_selector(state: ResearchState) -> str:
        return str(state.next_route)

    def build(self) -> Any:
        graph: StateGraph = StateGraph(ResearchState)
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("researcher", self._researcher_node)
        graph.add_node("analyst", self._analyst_node)
        graph.add_node("writer", self._writer_node)
        graph.add_node("critic", self._critic_node)

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._route_selector,
            {
                str(Route.RESEARCHER): "researcher",
                str(Route.ANALYST): "analyst",
                str(Route.WRITER): "writer",
                str(Route.CRITIC): "critic",
                str(Route.DONE): END,
            },
        )
        for worker in ("researcher", "analyst", "writer", "critic"):
            graph.add_edge(worker, "supervisor")
        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        compiled = self.build()
        result = compiled.invoke(state)
        if isinstance(result, ResearchState):
            return result
        return ResearchState.model_validate(result)


def build_default_workflow(settings: Settings) -> MultiAgentWorkflow:
    return MultiAgentWorkflow(
        llm_client=get_llm_client(settings),
        search_client=get_search_client(settings),
        max_iterations=settings.max_iterations,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_workflow.py -v`
Expected: PASS. If LangGraph recursion limit triggers, the supervisor's `max_iterations` should cap it; verify `out.iteration` increments per supervisor visit.

- [ ] **Step 5: Verify full offline suite + lint + types**

Run: `make lint && make typecheck && pytest -q`
Expected: PASS (remember `tests/test_agents_todo.py` still references stubs and may now fail — it is deleted in Task 14; if running before Task 14, expect that one file to fail).

- [ ] **Step 6: Commit** (confirm with user first)

```bash
git add src/multi_agent_research_lab/graph/workflow.py tests/test_workflow.py
git commit -m "feat(graph): implement LangGraph multi-agent workflow"
```

---

## Task 12: Scoring (heuristic + LLM-judge)

**Files:**
- Create: `src/multi_agent_research_lab/evaluation/scoring.py`
- Test: `tests/test_scoring.py` (create)

**Interfaces:**
- Consumes: `ResearchState`, `LLMClient`, `JUDGE_SYSTEM`, `judge_user`.
- Produces:
  - `heuristic_score(state: ResearchState, target_words: int = 300) -> float` — 0..10, deterministic. Components: length adequacy (answer word count vs target, capped), citation coverage (fraction of `state.sources` whose title appears in `final_answer`), non-empty notes presence. Weighted sum scaled to 0..10.
  - `llm_judge_score(state: ResearchState, llm_client: LLMClient) -> float` — parses `SCORE: <n>` from judge output; defaults to `5.0` if unparseable.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scoring.py`:

```python
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.scoring import heuristic_score, llm_judge_score
from multi_agent_research_lab.services.llm_client import FakeLLMClient


def _state_with_answer(answer: str) -> ResearchState:
    s = ResearchState(request=ResearchQuery(query="explain graphrag"))
    s.sources = [SourceDocument(title="GraphRAG paper", snippet="x")]
    s.research_notes = "n"
    s.analysis_notes = "a"
    s.final_answer = answer
    return s


def test_heuristic_score_is_deterministic_and_bounded():
    s = _state_with_answer("GraphRAG paper " + "word " * 300)
    score1 = heuristic_score(s)
    score2 = heuristic_score(s)
    assert score1 == score2
    assert 0.0 <= score1 <= 10.0


def test_heuristic_rewards_citation_presence():
    cited = heuristic_score(_state_with_answer("GraphRAG paper " + "w " * 300))
    uncited = heuristic_score(_state_with_answer("nothing relevant " + "w " * 300))
    assert cited > uncited


def test_llm_judge_defaults_when_unparseable():
    # FakeLLMClient never emits 'SCORE:' -> default 5.0
    assert llm_judge_score(_state_with_answer("ans"), FakeLLMClient()) == 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement scoring**

Create `evaluation/scoring.py`:

```python
"""Quality scoring: deterministic heuristic + optional LLM judge."""

import re

from multi_agent_research_lab.agents.prompts import JUDGE_SYSTEM, judge_user
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


def heuristic_score(state: ResearchState, target_words: int = 300) -> float:
    answer = state.final_answer or ""
    words = len(answer.split())
    length_ratio = min(words / target_words, 1.0) if target_words else 0.0

    sources = state.sources
    if sources:
        cited = sum(1 for s in sources if s.title and s.title in answer)
        citation_ratio = cited / len(sources)
    else:
        citation_ratio = 0.0

    notes_present = 1.0 if state.research_notes and state.analysis_notes else 0.0

    weighted = 0.5 * length_ratio + 0.3 * citation_ratio + 0.2 * notes_present
    return round(weighted * 10.0, 2)


def llm_judge_score(state: ResearchState, llm_client: LLMClient) -> float:
    resp = llm_client.complete(
        JUDGE_SYSTEM, judge_user(state.request.query, state.final_answer or "")
    )
    match = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", resp.content)
    if match is None:
        return 5.0
    return max(0.0, min(10.0, float(match.group(1))))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scoring.py -v`
Expected: PASS.

- [ ] **Step 5: Commit** (confirm with user first)

```bash
git add src/multi_agent_research_lab/evaluation/scoring.py tests/test_scoring.py
git commit -m "feat(eval): add heuristic and LLM-judge scoring"
```

---

## Task 13: Benchmark + report + storage

**Files:**
- Modify: `src/multi_agent_research_lab/evaluation/benchmark.py`
- Modify: `src/multi_agent_research_lab/evaluation/report.py`
- Modify: `src/multi_agent_research_lab/services/storage.py`
- Test: `tests/test_benchmark.py` (create), `tests/test_report.py` (extend)

**Interfaces:**
- Consumes: `ResearchState`, `BenchmarkMetrics`, `heuristic_score`, `LLMClient`.
- Produces:
  - `run_benchmark(run_name, query, runner, *, llm_client=None, judge=False) -> tuple[ResearchState, BenchmarkMetrics]` — measures latency, sums `cost_usd` across `state.agent_results` metadata into `estimated_cost_usd`, sets `quality_score = heuristic_score(state)` (or LLM-judge when `judge=True`).
  - `render_comparison(rows: list[BenchmarkMetrics]) -> str` (in `report.py`) — Markdown table.
  - `save_text(path: str, content: str) -> None` (in `storage.py`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_benchmark.py`:

```python
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.services.llm_client import FakeLLMClient
from multi_agent_research_lab.services.search_client import FakeSearchClient
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def test_run_benchmark_populates_metrics():
    def runner(query: str) -> ResearchState:
        wf = MultiAgentWorkflow(FakeLLMClient(), FakeSearchClient(), max_iterations=6)
        return wf.run(ResearchState(request=ResearchQuery(query=query, max_sources=2)))

    state, metrics = run_benchmark("multi", "explain graphrag", runner)
    assert metrics.run_name == "multi"
    assert metrics.latency_seconds >= 0
    assert metrics.quality_score is not None
    assert metrics.estimated_cost_usd is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark.py -v`
Expected: FAIL (signature mismatch / no quality population).

- [ ] **Step 3: Implement benchmark**

Replace `evaluation/benchmark.py`:

```python
"""Benchmark single-agent vs multi-agent runs."""

from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.scoring import heuristic_score, llm_judge_score
from multi_agent_research_lab.services.llm_client import LLMClient

Runner = Callable[[str], ResearchState]


def _sum_cost(state: ResearchState) -> float:
    total = 0.0
    for result in state.agent_results:
        cost = result.metadata.get("cost_usd")
        if isinstance(cost, (int, float)):
            total += float(cost)
    return total


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
    *,
    llm_client: LLMClient | None = None,
    judge: bool = False,
) -> tuple[ResearchState, BenchmarkMetrics]:
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    if judge and llm_client is not None:
        quality = llm_judge_score(state, llm_client)
    else:
        quality = heuristic_score(state)

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=_sum_cost(state),
        quality_score=quality,
    )
    return state, metrics
```

- [ ] **Step 4: Implement report rendering**

In `evaluation/report.py` add:

```python
from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_comparison(rows: list[BenchmarkMetrics]) -> str:
    header = "| Run | Latency (s) | Cost (USD) | Quality (0-10) |\n"
    sep = "|---|---:|---:|---:|\n"
    body = ""
    for r in rows:
        cost = f"{r.estimated_cost_usd:.4f}" if r.estimated_cost_usd is not None else "-"
        quality = f"{r.quality_score:.2f}" if r.quality_score is not None else "-"
        body += f"| {r.run_name} | {r.latency_seconds:.3f} | {cost} | {quality} |\n"
    return "# Benchmark Report\n\n" + header + sep + body
```

(If `report.py` already has content, append this function and keep existing exports.)

- [ ] **Step 5: Implement storage helper**

In `services/storage.py` add:

```python
from pathlib import Path


def save_text(path: str, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
```

- [ ] **Step 6: Extend report test**

In `tests/test_report.py` add:

```python
def test_render_comparison_outputs_markdown_table():
    from multi_agent_research_lab.core.schemas import BenchmarkMetrics
    from multi_agent_research_lab.evaluation.report import render_comparison
    rows = [
        BenchmarkMetrics(run_name="single", latency_seconds=1.0,
                         estimated_cost_usd=0.0, quality_score=5.0),
        BenchmarkMetrics(run_name="multi", latency_seconds=2.0,
                         estimated_cost_usd=0.0, quality_score=7.5),
    ]
    md = render_comparison(rows)
    assert "| single |" in md and "| multi |" in md
    assert "Quality" in md
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_benchmark.py tests/test_report.py -v`
Expected: PASS.

- [ ] **Step 8: Commit** (confirm with user first)

```bash
git add src/multi_agent_research_lab/evaluation/benchmark.py src/multi_agent_research_lab/evaluation/report.py src/multi_agent_research_lab/services/storage.py tests/test_benchmark.py tests/test_report.py
git commit -m "feat(eval): benchmark metrics, comparison report, storage helper"
```

---

## Task 14: CLI wiring (backend flag, real baseline, benchmark command)

**Files:**
- Modify: `src/multi_agent_research_lab/cli.py`
- Delete: `tests/test_agents_todo.py`
- Test: full suite.

**Interfaces:**
- Consumes: `get_settings`, `build_default_workflow`, `get_llm_client`, `run_benchmark`, `render_comparison`, `save_text`, `BASELINE_SYSTEM`.
- Produces CLI commands: `baseline`, `multi-agent`, `benchmark`, all accepting `--query/-q` and `--backend` (overrides `settings.llm_backend`).

- [ ] **Step 1: Delete the obsolete stub test**

Run: `git rm tests/test_agents_todo.py`

- [ ] **Step 2: Implement the CLI**

Replace `cli.py`:

```python
"""Command-line entrypoint for the lab."""

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.agents.prompts import BASELINE_SYSTEM
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_comparison
from multi_agent_research_lab.graph.workflow import build_default_workflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import get_llm_client
from multi_agent_research_lab.services.storage import save_text

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _settings(backend: str | None):
    settings = get_settings()
    configure_logging(settings.log_level)
    if backend:
        settings = settings.model_copy(update={"llm_backend": backend})
    return settings


def _run_baseline(query: str, settings) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    client = get_llm_client(settings)
    resp = client.complete(BASELINE_SYSTEM, query)
    state.final_answer = resp.content
    state.agent_results.append(
        AgentResult(agent=AgentName.WRITER, content=resp.content,
                    metadata={"cost_usd": resp.cost_usd})
    )
    return state


def _run_multi(query: str, settings) -> ResearchState:
    workflow = build_default_workflow(settings)
    return workflow.run(ResearchState(request=ResearchQuery(query=query)))


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q")],
    backend: Annotated[str | None, typer.Option("--backend")] = None,
) -> None:
    """Run a single-agent baseline."""
    settings = _settings(backend)
    state = _run_baseline(query, settings)
    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q")],
    backend: Annotated[str | None, typer.Option("--backend")] = None,
) -> None:
    """Run the multi-agent workflow."""
    settings = _settings(backend)
    state = _run_multi(query, settings)
    console.print(Panel.fit(state.final_answer or "", title="Multi-Agent"))
    console.print(f"Routes: {[str(r) for r in state.route_history]}")


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q")],
    backend: Annotated[str | None, typer.Option("--backend")] = None,
    out: Annotated[str, typer.Option("--out")] = "reports/benchmark_report.md",
) -> None:
    """Run single vs multi and write a comparison report."""
    settings = _settings(backend)
    _, single = run_benchmark("single", query, lambda q: _run_baseline(q, settings))
    _, multi = run_benchmark("multi", query, lambda q: _run_multi(q, settings))
    report = render_comparison([single, multi])
    save_text(out, report)
    console.print(report)
    console.print(f"\nSaved report to {out}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 3: Smoke-test the CLI offline**

Run: `python -m multi_agent_research_lab.cli benchmark -q "Research GraphRAG state-of-the-art"`
Expected: prints a Markdown table with `single` and `multi` rows; writes `reports/benchmark_report.md`; exit code 0.

- [ ] **Step 4: Run the full suite + lint + types**

Run: `make lint && make typecheck && pytest -q`
Expected: all PASS, no reference to `StudentTodoError` stubs remaining in tests.

- [ ] **Step 5: Commit** (confirm with user first)

```bash
git add src/multi_agent_research_lab/cli.py
git rm --cached tests/test_agents_todo.py 2>/dev/null || true
git commit -m "feat(cli): backend flag, real baseline, benchmark command; drop stub test"
```

---

## Task 15: Optional LangSmith tracing toggle

**Files:**
- Modify: `src/multi_agent_research_lab/observability/tracing.py`
- Test: `tests/test_tracing.py` (create)

**Interfaces:**
- Consumes: `Settings`.
- Produces: `configure_tracing(settings: Settings) -> bool` — when `settings.langsmith_api_key` is set, sets `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` env vars and returns `True`; otherwise returns `False` (no-op). Called from `_settings` in CLI.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tracing.py`:

```python
import os

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.observability.tracing import configure_tracing


def test_tracing_noop_without_key(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    assert configure_tracing(Settings(_env_file=None)) is False
    assert os.environ.get("LANGCHAIN_TRACING_V2") is None


def test_tracing_enables_with_key(monkeypatch):
    s = Settings(_env_file=None, LANGSMITH_API_KEY="k", LANGSMITH_PROJECT="p")
    assert configure_tracing(s) is True
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_PROJECT"] == "p"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tracing.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement tracing toggle**

In `observability/tracing.py` add:

```python
import os

from multi_agent_research_lab.core.config import Settings


def configure_tracing(settings: Settings) -> bool:
    """Enable LangSmith tracing when an API key is present. No-op otherwise."""
    if not settings.langsmith_api_key:
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    return True
```

- [ ] **Step 4: Wire it into the CLI**

In `cli.py` `_settings`, after `configure_logging(...)` add:

```python
    from multi_agent_research_lab.observability.tracing import configure_tracing
    configure_tracing(settings)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_tracing.py -v`
Expected: PASS.

- [ ] **Step 6: Final full verification**

Run: `make lint && make typecheck && pytest -q`
Expected: all PASS.

- [ ] **Step 7: Commit** (confirm with user first)

```bash
git add src/multi_agent_research_lab/observability/tracing.py src/multi_agent_research_lab/cli.py tests/test_tracing.py
git commit -m "feat(observability): optional LangSmith tracing toggle"
```

---

## Self-Review

**Spec coverage:**
- Backend abstraction (LLM/search fake+real+factory) → Tasks 3, 4. ✅
- OpenRouter via openai SDK → Task 3. ✅
- LangGraph StateGraph orchestration → Task 11. ✅
- 5 agents incl. Critic + bounded revision loop → Tasks 6–10. ✅
- Supervisor routing policy with max_iterations → Task 10. ✅
- In-state trace (always) → every agent `add_trace_event`; LangSmith optional → Task 15. ✅
- Benchmark heuristic always + LLM-judge optional + cost from tokens → Tasks 12, 13. ✅
- Single-agent baseline real → Task 14. ✅
- CLI `--backend` + `benchmark` command writing `reports/benchmark_report.md` → Task 14. ✅
- Config OpenRouter replacement + `.env.example` → Task 1. ✅
- Deps: LangGraph to core, openai/tavily optional → Task 1. ✅
- Replace `test_agents_todo.py` with behavior tests → Tasks 6–11, deletion in Task 14. ✅
- Error handling: tenacity retry + `AgentExecutionError` normalization → Tasks 3, 4. ✅
- Storage writes reports → Task 13. ✅

**Placeholder scan:** No TBD/TODO; all code steps contain full code. ✅

**Type consistency:** `LLMClient.complete(system_prompt, user_prompt) -> LLMResponse`, `SearchClient.search(query, max_results) -> list[SourceDocument]`, `SupervisorAgent.decide -> Route`, `state.next_route: str | None`, `_verdict_after_last_write` used consistently in Task 10. `run_benchmark(..., *, llm_client, judge)` signature consistent between Tasks 13 and 14. ✅

**Known risk flagged:** LangGraph node return convention — nodes return dicts of real model objects via `_updates`; `run()` coerces the result back to `ResearchState` via `model_validate` if LangGraph returns a dict. Verify in Task 11 Step 4.
