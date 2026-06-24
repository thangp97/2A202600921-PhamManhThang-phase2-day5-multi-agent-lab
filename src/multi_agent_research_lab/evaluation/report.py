"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown.

    TODO(student): Add richer analysis, examples, screenshots, and trace links.
    """

    header = "# Benchmark Report"
    table_header = "| Run | Latency (s) | Cost (USD) | Quality | Notes |"
    table_sep = "|---|---:|---:|---:|---|"
    lines = [header, "", table_header, table_sep]
    for item in metrics:
        cost = (
            "" if item.estimated_cost_usd is None
            else f"{item.estimated_cost_usd:.4f}"
        )
        quality = (
            "" if item.quality_score is None else f"{item.quality_score:.1f}"
        )
        row = (
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | "
            f"{quality} | {item.notes} |"
        )
        lines.append(row)
    return "\n".join(lines) + "\n"


def render_comparison(rows: list[BenchmarkMetrics]) -> str:
    """Render a comparison table of multiple benchmark runs."""
    header = "| Run | Latency (s) | Cost (USD) | Quality (0-10) |\n"
    sep = "|---|---:|---:|---:|\n"
    body = ""
    for r in rows:
        cost = f"{r.estimated_cost_usd:.4f}" if r.estimated_cost_usd is not None else "-"
        quality = f"{r.quality_score:.2f}" if r.quality_score is not None else "-"
        body += f"| {r.run_name} | {r.latency_seconds:.3f} | {cost} | {quality} |\n"
    return "# Benchmark Report\n\n" + header + sep + body
