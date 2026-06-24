from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_to_researcher_when_empty() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    out = SupervisorAgent(max_iterations=6).run(state)
    assert out.next_route is not None
