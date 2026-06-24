"""Command-line entrypoint for the lab."""

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.agents.prompts import BASELINE_SYSTEM
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_comparison
from multi_agent_research_lab.graph.workflow import build_default_workflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import get_llm_client
from multi_agent_research_lab.services.storage import save_text

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _settings(backend: str | None) -> Settings:
    settings = get_settings()
    configure_logging(settings.log_level)
    if backend:
        settings = settings.model_copy(update={"llm_backend": backend})
    return settings


def _run_baseline(query: str, settings: Settings) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    client = get_llm_client(settings)
    resp = client.complete(BASELINE_SYSTEM, query)
    state.final_answer = resp.content
    state.agent_results.append(
        AgentResult(agent=AgentName.WRITER, content=resp.content,
                    metadata={"cost_usd": resp.cost_usd})
    )
    return state


def _run_multi(query: str, settings: Settings) -> ResearchState:
    workflow = build_default_workflow(settings)
    return workflow.run(ResearchState(request=ResearchQuery(query=query)))


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q")],
    backend: Annotated[str | None, typer.Option("--backend")] = None,
) -> None:
    """Run a single-agent baseline."""
    settings = _settings(backend)
    state = _run_baseline(query, settings)
    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q")],
    backend: Annotated[str | None, typer.Option("--backend")] = None,
) -> None:
    """Run the multi-agent workflow."""
    settings = _settings(backend)
    state = _run_multi(query, settings)
    console.print(Panel.fit(state.final_answer or "", title="Multi-Agent"))
    console.print(f"Routes: {[str(r) for r in state.route_history]}")


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q")],
    backend: Annotated[str | None, typer.Option("--backend")] = None,
    out: Annotated[str, typer.Option("--out")] = "reports/benchmark_report.md",
) -> None:
    """Run single vs multi and write a comparison report."""
    settings = _settings(backend)
    _, single = run_benchmark("single", query, lambda q: _run_baseline(q, settings))
    _, multi = run_benchmark("multi", query, lambda q: _run_multi(q, settings))
    report = render_comparison([single, multi])
    save_text(out, report)
    console.print(report)
    console.print(f"\nSaved report to {out}")


if __name__ == "__main__":
    app()
