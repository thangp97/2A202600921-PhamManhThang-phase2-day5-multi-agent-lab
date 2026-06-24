"""Tracing hooks.

Provider-agnostic by design: an in-process span by default, augmented with a
LangSmith run when tracing is enabled and the SDK is installed. Students can
swap LangSmith for Langfuse/OpenTelemetry by editing `_provider_span` only.
"""

import importlib
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import Settings

logger = logging.getLogger("multi_agent_research_lab.tracing")


def _provider_span(name: str, attributes: dict[str, Any] | None) -> Any:
    """Return a LangSmith span context manager, or None when unavailable.

    No-ops (returns None) unless tracing was enabled by `configure_tracing` and
    the optional `langsmith` package can be imported. Never raises.
    """
    if os.environ.get("LANGCHAIN_TRACING_V2") != "true":
        return None
    try:
        module = importlib.import_module("langsmith")  # lazy: optional dependency
    except Exception:
        return None
    trace = getattr(module, "trace", None)
    if trace is None:
        return None
    try:
        return trace(name=name, run_type="chain", inputs=attributes or {})
    except Exception:
        return None


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Time a unit of work as a span.

    Always yields an in-process span dict (`name`, `attributes`,
    `duration_seconds`); additionally emits a LangSmith run when tracing is on.
    """

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    provider = _provider_span(name, attributes)
    try:
        if provider is None:
            yield span
        else:
            with provider as run:
                span["provider_run"] = run
                yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
        logger.debug("span '%s' finished in %.4fs", name, span["duration_seconds"])


def configure_tracing(settings: Settings) -> bool:
    """Enable LangSmith tracing when an API key is present. No-op otherwise."""
    if not settings.langsmith_api_key:
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    return True
