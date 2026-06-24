"""Researcher agent: gathers sources and writes research notes."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.prompts import RESEARCHER_SYSTEM
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, llm_client: LLMClient, search_client: SearchClient) -> None:
        self._llm = llm_client
        self._search = search_client

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        state.sources = self._search.search(
            state.request.query, max_results=state.request.max_sources
        )
        sources_text = "\n".join(f"- {s.title}: {s.snippet}" for s in state.sources)
        user = f"Query: {state.request.query}\n\nSources:\n{sources_text}"
        resp = self._llm.complete(RESEARCHER_SYSTEM, user)
        state.research_notes = resp.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=resp.content,
                metadata={
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cost_usd": resp.cost_usd,
                    "num_sources": len(state.sources),
                },
            )
        )
        state.add_trace_event("researcher", {"num_sources": len(state.sources)})
        return state
