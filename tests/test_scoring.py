from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.scoring import heuristic_score, llm_judge_score
from multi_agent_research_lab.services.llm_client import FakeLLMClient


def _state_with_answer(answer: str) -> ResearchState:
    s = ResearchState(request=ResearchQuery(query="explain graphrag"))
    s.sources = [SourceDocument(title="GraphRAG paper", snippet="x")]
    s.research_notes = "n"
    s.analysis_notes = "a"
    s.final_answer = answer
    return s


def test_heuristic_score_is_deterministic_and_bounded():
    s = _state_with_answer("GraphRAG paper " + "word " * 300)
    score1 = heuristic_score(s)
    score2 = heuristic_score(s)
    assert score1 == score2
    assert 0.0 <= score1 <= 10.0


def test_heuristic_rewards_citation_presence():
    cited = heuristic_score(_state_with_answer("GraphRAG paper " + "w " * 300))
    uncited = heuristic_score(_state_with_answer("nothing relevant " + "w " * 300))
    assert cited > uncited


def test_llm_judge_defaults_when_unparseable():
    # FakeLLMClient never emits 'SCORE:' -> default 5.0
    assert llm_judge_score(_state_with_answer("ans"), FakeLLMClient()) == 5.0
