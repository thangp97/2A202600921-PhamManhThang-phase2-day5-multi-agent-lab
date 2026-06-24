"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to a detailed markdown table."""

    header = "# Benchmark Report"
    table_header = (
        "| Run | Latency (s) | Tokens | Cost (USD) | Quality "
        "| Citation cov. | Failure rate | Notes |"
    )
    table_sep = "|---|---:|---:|---:|---:|---:|---:|---|"
    lines = [header, "", table_header, table_sep]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        tokens = "" if item.total_tokens is None else f"{item.total_tokens}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        cov = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        fail = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        row = (
            f"| {item.run_name} | {item.latency_seconds:.2f} | {tokens} | {cost} "
            f"| {quality} | {cov} | {fail} | {item.notes} |"
        )
        lines.append(row)
    return "\n".join(lines) + "\n"


def render_comparison(rows: list[BenchmarkMetrics]) -> str:
    """Render a comparison table of multiple benchmark runs."""
    header = (
        "| Run | Latency (s) | Tokens | Cost (USD) | Quality (0-10) "
        "| Citation cov. | Failure rate |\n"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|\n"
    body = ""
    for r in rows:
        cost = f"{r.estimated_cost_usd:.4f}" if r.estimated_cost_usd is not None else "-"
        tokens = f"{r.total_tokens}" if r.total_tokens is not None else "-"
        quality = f"{r.quality_score:.2f}" if r.quality_score is not None else "-"
        cov = f"{r.citation_coverage:.0%}" if r.citation_coverage is not None else "-"
        fail = f"{r.failure_rate:.0%}" if r.failure_rate is not None else "-"
        body += (
            f"| {r.run_name} | {r.latency_seconds:.3f} | {tokens} | {cost} "
            f"| {quality} | {cov} | {fail} |\n"
        )
    return "# Benchmark Report\n\n" + header + sep + body
