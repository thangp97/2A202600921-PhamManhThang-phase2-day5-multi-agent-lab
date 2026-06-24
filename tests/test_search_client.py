from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.services.search_client import (
    FakeSearchClient,
    get_search_client,
)


def test_fake_search_returns_n_sources() -> None:
    client = FakeSearchClient()
    docs = client.search("graphrag", max_results=3)
    assert len(docs) == 3
    assert all(isinstance(d, SourceDocument) for d in docs)
    assert docs[0].title == "Fake source 0 for graphrag"


def test_factory_returns_fake_by_default() -> None:
    assert isinstance(get_search_client(Settings()), FakeSearchClient)
