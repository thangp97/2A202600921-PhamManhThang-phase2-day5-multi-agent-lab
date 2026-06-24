from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.services.llm_client import (
    FakeLLMClient,
    LLMResponse,
    get_llm_client,
)


def test_fake_llm_is_deterministic() -> None:
    client = FakeLLMClient()
    r1 = client.complete("sysA", "userA")
    r2 = client.complete("sysA", "userA")
    assert isinstance(r1, LLMResponse)
    assert r1.content == r2.content
    assert r1.input_tokens == 10 and r1.output_tokens == 20


def test_factory_returns_fake_by_default() -> None:
    settings = Settings()
    assert isinstance(get_llm_client(settings), FakeLLMClient)
