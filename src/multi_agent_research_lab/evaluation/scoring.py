"""Quality scoring: deterministic heuristic + optional LLM judge."""

import re

from multi_agent_research_lab.agents.prompts import JUDGE_SYSTEM, judge_user
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


def heuristic_score(state: ResearchState, target_words: int = 300) -> float:
    answer = state.final_answer or ""
    words = len(answer.split())
    length_ratio = min(words / target_words, 1.0) if target_words else 0.0

    sources = state.sources
    if sources:
        cited = sum(1 for s in sources if s.title and s.title in answer)
        citation_ratio = cited / len(sources)
    else:
        citation_ratio = 0.0

    notes_present = 1.0 if state.research_notes and state.analysis_notes else 0.0

    weighted = 0.5 * length_ratio + 0.3 * citation_ratio + 0.2 * notes_present
    return round(weighted * 10.0, 2)


def llm_judge_score(state: ResearchState, llm_client: LLMClient) -> float:
    resp = llm_client.complete(
        JUDGE_SYSTEM, judge_user(state.request.query, state.final_answer or "")
    )
    match = re.search(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", resp.content)
    if match is None:
        return 5.0
    return max(0.0, min(10.0, float(match.group(1))))
