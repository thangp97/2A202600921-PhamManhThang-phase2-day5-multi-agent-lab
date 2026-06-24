from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.evaluation.report import render_markdown_report


def test_report_renders_markdown() -> None:
    report = render_markdown_report([BenchmarkMetrics(run_name="baseline", latency_seconds=1.23)])
    assert "Benchmark Report" in report
    assert "baseline" in report


def test_render_comparison_outputs_markdown_table():
    from multi_agent_research_lab.evaluation.report import render_comparison

    rows = [
        BenchmarkMetrics(run_name="single", latency_seconds=1.0,
                         estimated_cost_usd=0.0, quality_score=5.0),
        BenchmarkMetrics(run_name="multi", latency_seconds=2.0,
                         estimated_cost_usd=0.0, quality_score=7.5),
    ]
    md = render_comparison(rows)
    assert "| single |" in md and "| multi |" in md
    assert "Quality" in md
