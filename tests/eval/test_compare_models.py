# tests/eval/test_compare_models.py
import json

from llm_internal.eval.benchmark import BenchmarkCase
from llm_internal.eval.compare_models import (
    ComparisonReport,
    compare_metrics,
    compare_reports,
    comparison_to_json,
    format_comparison,
)
from llm_internal.eval.metrics import score_benchmark

_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def _cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            id="c1", category="single_tool_selection", description="d", tools=[_TOOL],
            messages=[{"role": "user", "content": "weather in Paris?"}],
            expects_tool_call=True, expected_tool_calls=[{"name": "get_weather", "arguments": {"city": "Paris"}}],
        ),
        BenchmarkCase(
            id="c2", category="no_tool_plain_chat", description="d", tools=[],
            messages=[{"role": "user", "content": "hi"}], expects_tool_call=False,
        ),
    ]


def test_compare_metrics_flags_improvement_when_fine_tuned_exceeds_base():
    base = score_benchmark(_cases(), ["I don't know", "Hi there!"]).overall
    fine_tuned = score_benchmark(
        _cases(),
        ['<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>', "Hi there!"],
    ).overall

    comparisons = compare_metrics(base, fine_tuned)

    selection = next(c for c in comparisons if c.metric == "tool_selection_accuracy")
    assert selection.improvement is True
    assert selection.regression is False
    assert selection.delta > 0


def test_compare_metrics_flags_regression_when_fine_tuned_underperforms():
    base = score_benchmark(
        _cases(),
        ['<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>', "Hi there!"],
    ).overall
    fine_tuned = score_benchmark(_cases(), ["I don't know", "Hi there!"]).overall

    comparisons = compare_metrics(base, fine_tuned)

    selection = next(c for c in comparisons if c.metric == "tool_selection_accuracy")
    assert selection.regression is True
    assert selection.improvement is False


def test_compare_metrics_ignores_noise_below_epsilon():
    base = score_benchmark(_cases(), ["I don't know", "Hi there!"]).overall
    fine_tuned = base  # identical -> zero delta everywhere

    comparisons = compare_metrics(base, fine_tuned)

    assert all(not c.regression and not c.improvement for c in comparisons)


def test_compare_reports_collects_regressions_and_improvements_and_gate_verdicts():
    base_report = score_benchmark(_cases(), ["I don't know", "Hi there!"])
    ft_report = score_benchmark(
        _cases(),
        ['<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>', "Hi there!"],
    )

    comparison = compare_reports(base_report, ft_report)

    assert isinstance(comparison, ComparisonReport)
    assert "tool_selection_accuracy" in comparison.improvements
    assert comparison.regressions == []
    assert comparison.fine_tuned_gates_passed in (True, False)  # exercised, not GPU-dependent here


def test_format_comparison_reports_regressions_and_improvements_sections():
    base_report = score_benchmark(_cases(), ["I don't know", "Hi there!"])
    ft_report = score_benchmark(
        _cases(),
        ['<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>', "Hi there!"],
    )
    comparison = compare_reports(base_report, ft_report)

    text = format_comparison(comparison)

    assert "regressions:" in text
    assert "improvements:" in text
    assert "tool_selection_accuracy" in text


def test_comparison_to_json_is_serializable():
    base_report = score_benchmark(_cases(), ["I don't know", "Hi there!"])
    ft_report = score_benchmark(_cases(), ["I don't know", "Hi there!"])
    comparison = compare_reports(base_report, ft_report)

    payload = comparison_to_json(comparison)
    serialized = json.dumps(payload)

    reloaded = json.loads(serialized)
    assert reloaded["regressions"] == []
    assert reloaded["improvements"] == []
