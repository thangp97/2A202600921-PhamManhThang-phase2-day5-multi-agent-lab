"""Benchmark single-agent vs multi-agent runs."""

from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.scoring import (
    citation_coverage,
    heuristic_score,
    llm_judge_score,
)
from multi_agent_research_lab.services.llm_client import LLMClient

Runner = Callable[[str], ResearchState]


def _sum_cost(state: ResearchState) -> float | None:
    """Tổng cost USD nếu provider trả về; None nếu không có dữ liệu cost."""
    total = 0.0
    found = False
    for result in state.agent_results:
        cost = result.metadata.get("cost_usd")
        if isinstance(cost, (int, float)):
            total += float(cost)
            found = True
    return total if found else None


def _sum_tokens(state: ResearchState) -> int | None:
    """Tổng input + output tokens trên mọi lượt agent (proxy cho cost)."""
    total = 0
    found = False
    for result in state.agent_results:
        for key in ("input_tokens", "output_tokens"):
            value = result.metadata.get(key)
            if isinstance(value, int):
                total += value
                found = True
    return total if found else None


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
    *,
    llm_client: LLMClient | None = None,
    judge: bool = False,
) -> tuple[ResearchState, BenchmarkMetrics]:
    started = perf_counter()
    failed = False
    try:
        state = runner(query)
    except Exception as exc:  # benchmark ghi nhận lỗi thay vì crash
        failed = True
        state = ResearchState(request=ResearchQuery(query=query))
        state.errors.append(str(exc))
    latency = perf_counter() - started

    quality: float | None
    if failed:
        quality = 0.0
    elif judge and llm_client is not None:
        quality = llm_judge_score(state, llm_client)
    else:
        quality = heuristic_score(state)

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=_sum_cost(state),
        total_tokens=_sum_tokens(state),
        quality_score=quality,
        citation_coverage=citation_coverage(state),
        failure_rate=1.0 if failed else 0.0,
    )
    return state, metrics
