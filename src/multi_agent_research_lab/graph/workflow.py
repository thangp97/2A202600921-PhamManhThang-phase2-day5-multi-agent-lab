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
        graph: StateGraph[ResearchState] = StateGraph(ResearchState)
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
