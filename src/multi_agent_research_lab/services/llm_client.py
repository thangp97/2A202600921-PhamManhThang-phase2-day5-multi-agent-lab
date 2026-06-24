"""LLM client abstraction.

Agents depend on this interface instead of importing an SDK directly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import AgentExecutionError


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion."""


class FakeLLMClient(LLMClient):
    """Deterministic offline client for tests and no-key runs."""

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        content = f"[fake-llm] sys={system_prompt[:32]} | user={user_prompt[:64]}"
        return LLMResponse(
            content=content, input_tokens=10, output_tokens=20, cost_usd=0.0
        )


class OpenRouterLLMClient(LLMClient):
    """Real backend via the OpenAI SDK pointed at OpenRouter."""

    def __init__(
        self, api_key: str, base_url: str, model: str, timeout: int
    ) -> None:
        from openai import OpenAI  # lazy import: optional dependency

        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self._model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - normalize to domain error
            raise AgentExecutionError(f"OpenRouter call failed: {exc}") from exc
        usage = resp.usage
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            cost_usd=None,
        )


def get_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_backend == "openrouter":
        if not settings.openrouter_api_key:
            raise AgentExecutionError(
                "OPENROUTER_API_KEY required for openrouter backend"
            )
        return OpenRouterLLMClient(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            model=settings.openrouter_model,
            timeout=settings.timeout_seconds,
        )
    return FakeLLMClient()
