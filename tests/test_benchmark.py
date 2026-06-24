from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import FakeLLMClient
from multi_agent_research_lab.services.search_client import FakeSearchClient


def test_run_benchmark_populates_metrics():
    def runner(query: str) -> ResearchState:
        wf = MultiAgentWorkflow(FakeLLMClient(), FakeSearchClient(), max_iterations=6)
        return wf.run(ResearchState(request=ResearchQuery(query=query, max_sources=2)))

    state, metrics = run_benchmark("multi", "explain graphrag", runner)
    assert metrics.run_name == "multi"
    assert metrics.latency_seconds >= 0
    assert metrics.quality_score is not None
    assert metrics.estimated_cost_usd is not None
