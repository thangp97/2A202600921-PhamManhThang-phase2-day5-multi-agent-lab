"""Supervisor / router: decides the next agent and the stop condition."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, Route, Verdict
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    name = "supervisor"

    def __init__(self, max_iterations: int) -> None:
        self._max_iterations = max_iterations

    def _verdict_after_last_write(self, state: ResearchState) -> str | None:
        last_writer_idx = -1
        for i, r in enumerate(state.agent_results):
            if r.agent == AgentName.WRITER:
                last_writer_idx = i
        for r in state.agent_results[last_writer_idx + 1 :]:
            if r.agent == AgentName.CRITIC:
                return r.metadata.get("verdict")
        return None

    def decide(self, state: ResearchState) -> Route:
        if state.iteration >= self._max_iterations:
            return Route.DONE
        verdict = self._verdict_after_last_write(state)
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
