from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, Verdict
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import FakeLLMClient


def test_parse_critique_reads_score_and_verdict():
    agent = CriticAgent(FakeLLMClient())
    score, verdict, feedback = agent.parse_critique(
        "SCORE: 8\nVERDICT: accept\nLooks complete."
    )
    assert score == 8.0
    assert verdict == Verdict.ACCEPT
    assert "complete" in feedback


def test_parse_critique_defaults_on_garbage():
    agent = CriticAgent(FakeLLMClient(), accept_threshold=7.0)
    score, verdict, _ = agent.parse_critique("no structured output here")
    assert verdict == Verdict.ACCEPT
    assert score == 7.0


def test_critic_run_records_quality_score():
    state = ResearchState(request=ResearchQuery(query="explain graphrag"))
    state.final_answer = "an answer"
    out = CriticAgent(FakeLLMClient()).run(state)
    last = out.agent_results[-1]
    assert last.agent == AgentName.CRITIC
    assert last.quality_score is not None
    assert "verdict" in last.metadata
