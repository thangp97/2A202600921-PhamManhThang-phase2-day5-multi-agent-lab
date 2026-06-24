"""Critic agent: scores the final answer and emits an accept/revise verdict."""

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.prompts import CRITIC_SYSTEM, critic_user
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, Verdict
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class CriticAgent(BaseAgent):
    """Scores the final answer and emits an accept/revise verdict."""

    name = "critic"

    def __init__(self, llm_client: LLMClient, accept_threshold: float = 7.0) -> None:
        self._llm = llm_client
        self._threshold = accept_threshold

    def parse_critique(self, text: str) -> tuple[float, str, str]:
        """Parse critique response to extract score, verdict, and feedback.

        Returns (score, verdict, feedback). On parse failure, defaults to
        (accept_threshold, Verdict.ACCEPT, text).
        """
        score_match = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", text)
        verdict_match = re.search(r"VERDICT:\s*(accept|revise)", text, re.IGNORECASE)
        if score_match is None and verdict_match is None:
            return self._threshold, Verdict.ACCEPT, text
        score = float(score_match.group(1)) if score_match else self._threshold
        score = max(0.0, min(10.0, score))
        if verdict_match:
            verdict = Verdict(verdict_match.group(1).lower())
        else:
            verdict = Verdict.ACCEPT if score >= self._threshold else Verdict.REVISE
        return score, verdict, text

    def run(self, state: ResearchState) -> ResearchState:
        """Score final answer and append critique to agent results."""
        resp = self._llm.complete(CRITIC_SYSTEM, critic_user(state.final_answer or ""))
        score, verdict, feedback = self.parse_critique(resp.content)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=resp.content,
                quality_score=score,
                metadata={"verdict": str(verdict), "feedback": feedback},
            )
        )
        state.add_trace_event("critic", {"score": score, "verdict": str(verdict)})
        return state
