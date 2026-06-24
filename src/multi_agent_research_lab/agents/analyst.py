"""Analyst agent: turns research notes into structured insights."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.prompts import ANALYST_SYSTEM
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`.

        Extract key claims, compare viewpoints, and flag weak evidence.
        """
        user = f"Research notes:\n{state.research_notes or ''}"
        resp = self._llm.complete(ANALYST_SYSTEM, user)
        state.analysis_notes = resp.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=resp.content,
                metadata={
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cost_usd": resp.cost_usd,
                },
            )
        )
        state.add_trace_event("analyst", {})
        return state
