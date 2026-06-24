"""Tests for the LangGraph multi-agent workflow."""

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import FakeLLMClient
from multi_agent_research_lab.services.search_client import FakeSearchClient


def test_workflow_runs_to_completion_offline() -> None:
    wf = MultiAgentWorkflow(FakeLLMClient(), FakeSearchClient(), max_iterations=6)
    state = ResearchState(request=ResearchQuery(query="explain graphrag", max_sources=2))
    out = wf.run(state)
    assert out.final_answer is not None and out.final_answer != ""
    assert out.research_notes is not None
    assert out.analysis_notes is not None
    # fake critic accepts on first pass -> terminates without exhausting iterations
    assert out.iteration <= 6
    assert "done" in [str(r) for r in out.route_history]


def test_workflow_build_returns_compiled_graph() -> None:
    wf = MultiAgentWorkflow(FakeLLMClient(), FakeSearchClient(), max_iterations=6)
    compiled = wf.build()
    assert compiled is not None


def test_build_default_workflow() -> None:
    from multi_agent_research_lab.core.config import Settings
    from multi_agent_research_lab.graph.workflow import build_default_workflow

    settings = Settings()
    wf = build_default_workflow(settings)
    assert isinstance(wf, MultiAgentWorkflow)


def test_workflow_max_iterations_terminates() -> None:
    """Even with max_iterations=4, workflow must terminate (not hang).

    iteration counts every supervisor visit including the final 'done' routing,
    so the cap is max_iterations + 1 at most (normal path: researcher, analyst,
    writer, critic, done = 5 supervisor calls with max_iterations=4).
    """
    wf = MultiAgentWorkflow(FakeLLMClient(), FakeSearchClient(), max_iterations=4)
    state = ResearchState(request=ResearchQuery(query="explain graphrag", max_sources=1))
    out = wf.run(state)
    # +1 because the final "done" routing is one extra supervisor visit
    assert out.iteration <= 5
    assert "done" in [str(r) for r in out.route_history]
