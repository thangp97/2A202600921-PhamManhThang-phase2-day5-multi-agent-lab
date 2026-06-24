from multi_agent_research_lab.core.config import Settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.openrouter_model
    assert settings.max_iterations >= 1


def test_settings_defaults_to_fake_backends(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("SEARCH_BACKEND", raising=False)
    s = Settings(_env_file=None)
    assert s.llm_backend == "fake"
    assert s.search_backend == "fake"
    assert s.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert s.openrouter_model == "openai/gpt-4o-mini"
