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
