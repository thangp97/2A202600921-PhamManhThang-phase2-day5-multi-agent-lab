"""Writer agent: composes the final answer."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.prompts import WRITER_SYSTEM
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer` and record the result."""
        user = (
            f"Audience: {state.request.audience}\n"
            f"Query: {state.request.query}\n\n"
            f"Research notes:\n{state.research_notes or ''}\n\n"
            f"Analysis notes:\n{state.analysis_notes or ''}"
        )
        resp = self._llm.complete(WRITER_SYSTEM, user)
        state.final_answer = resp.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=resp.content,
                metadata={
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cost_usd": resp.cost_usd,
                },
            )
        )
        state.add_trace_event("writer", {})
        state.add_trace_event("writer_revision_marker", {"answer_version": len(
            [r for r in state.agent_results if r.agent == AgentName.WRITER])})
        return state
