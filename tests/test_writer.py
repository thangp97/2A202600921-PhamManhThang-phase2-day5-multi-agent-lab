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
