from multi_agent_research_lab.core.schemas import ResearchQuery, Route
from multi_agent_research_lab.core.state import ResearchState


def test_state_records_route_and_trace() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.record_route("researcher")
    state.add_trace_event("route", {"next": "researcher"})
    assert state.iteration == 1
    assert state.route_history == ["researcher"]
    assert state.trace[0]["name"] == "route"


def test_state_has_next_route_and_route_enum() -> None:
    state = ResearchState(request=ResearchQuery(query="hello world"))
    assert state.next_route is None
    state.next_route = Route.RESEARCHER
    assert state.next_route == "researcher"
