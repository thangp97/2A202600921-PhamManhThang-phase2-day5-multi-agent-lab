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
    s.agent_results.append(AgentResult(agent=AgentName.WRITER, content="ans"))
    s.agent_results.append(
        AgentResult(agent=AgentName.CRITIC, content="ok", quality_score=9,
                    metadata={"verdict": str(Verdict.ACCEPT)})
    )
    assert SupervisorAgent(max_iterations=6).decide(s) == Route.DONE


def test_revise_routes_back_to_writer():
    s = _state()
    s.research_notes, s.analysis_notes, s.final_answer = "n", "a", "ans"
    s.agent_results.append(AgentResult(agent=AgentName.WRITER, content="ans"))
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
