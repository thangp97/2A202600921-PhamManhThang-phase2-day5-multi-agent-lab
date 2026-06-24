"""Search client abstraction (provider-agnostic)."""

from abc import ABC, abstractmethod
from typing import Any

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Return source documents for a query."""


class FakeSearchClient(SearchClient):
    def search(self, query: str, max_results: int) -> list[SourceDocument]:
        return [
            SourceDocument(
                title=f"Fake source {i} for {query}",
                url=f"https://example.test/{i}",
                snippet=f"Deterministic snippet {i} about {query}.",
            )
            for i in range(max_results)
        ]


class TavilySearchClient(SearchClient):
    _client: Any

    def __init__(self, api_key: str) -> None:
        import tavily  # type: ignore[import-not-found]

        self._client = tavily.TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int) -> list[SourceDocument]:
        try:
            res = self._client.search(query=query, max_results=max_results)
        except Exception as exc:  # noqa: BLE001
            raise AgentExecutionError(f"Tavily search failed: {exc}") from exc
        return [
            SourceDocument(
                title=item.get("title", "Untitled"),
                url=item.get("url"),
                snippet=item.get("content", ""),
            )
            for item in res.get("results", [])
        ]


def get_search_client(settings: Settings) -> SearchClient:
    if settings.search_backend == "tavily":
        if not settings.tavily_api_key:
            raise AgentExecutionError("TAVILY_API_KEY required for tavily backend")
        return TavilySearchClient(api_key=settings.tavily_api_key)
    return FakeSearchClient()
