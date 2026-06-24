import os

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.observability.tracing import configure_tracing


def test_tracing_noop_without_key(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    assert configure_tracing(Settings(_env_file=None)) is False
    assert os.environ.get("LANGCHAIN_TRACING_V2") is None


def test_tracing_enables_with_key(monkeypatch):
    s = Settings(_env_file=None, LANGSMITH_API_KEY="k", LANGSMITH_PROJECT="p")
    assert configure_tracing(s) is True
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_PROJECT"] == "p"
