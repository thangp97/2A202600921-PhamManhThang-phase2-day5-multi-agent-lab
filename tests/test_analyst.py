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
