"""Benchmark single-agent vs multi-agent runs."""

from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.scoring import heuristic_score, llm_judge_score
from multi_agent_research_lab.services.llm_client import LLMClient

Runner = Callable[[str], ResearchState]


def _sum_cost(state: ResearchState) -> float:
    total = 0.0
    for result in state.agent_results:
        cost = result.metadata.get("cost_usd")
        if isinstance(cost, (int, float)):
            total += float(cost)
    return total


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
    *,
    llm_client: LLMClient | None = None,
    judge: bool = False,
) -> tuple[ResearchState, BenchmarkMetrics]:
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    if judge and llm_client is not None:
        quality = llm_judge_score(state, llm_client)
    else:
        quality = heuristic_score(state)

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=_sum_cost(state),
        quality_score=quality,
    )
    return state, metrics
