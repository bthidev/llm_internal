"""Base-model vs. fine-tuned-model comparison on the independent benchmark."""

from __future__ import annotations

import dataclasses
import json
import sys

from llm_internal.eval.benchmark import load_benchmark
from llm_internal.eval.config import load_benchmark_eval_config
from llm_internal.eval.gates import (
    DEFAULT_GATES,
    HIGHER_IS_BETTER,
    METRIC_DIRECTIONS,
    apply_overrides,
    evaluate_gates,
    gates_passed,
)
from llm_internal.eval.generation import generate_for_messages
from llm_internal.eval.metrics import BenchmarkMetrics, BenchmarkReport, score_benchmark

_REGRESSION_EPSILON = 0.01


@dataclasses.dataclass(frozen=True)
class MetricComparison:
    metric: str
    base: float
    fine_tuned: float
    delta: float
    regression: bool
    improvement: bool


@dataclasses.dataclass(frozen=True)
class ComparisonReport:
    base: BenchmarkReport
    fine_tuned: BenchmarkReport
    metric_comparisons: list[MetricComparison]
    regressions: list[str]
    improvements: list[str]
    base_gates_passed: bool
    fine_tuned_gates_passed: bool


def _compare_metric(name: str, base_value: float, ft_value: float) -> MetricComparison:
    delta = ft_value - base_value
    direction = METRIC_DIRECTIONS[name]
    if direction == HIGHER_IS_BETTER:
        regression = delta < -_REGRESSION_EPSILON
        improvement = delta > _REGRESSION_EPSILON
    else:
        regression = delta > _REGRESSION_EPSILON
        improvement = delta < -_REGRESSION_EPSILON
    return MetricComparison(
        metric=name,
        base=base_value,
        fine_tuned=ft_value,
        delta=delta,
        regression=regression,
        improvement=improvement,
    )


def compare_metrics(base: BenchmarkMetrics, fine_tuned: BenchmarkMetrics) -> list[MetricComparison]:
    base_dict, ft_dict = base.as_dict(), fine_tuned.as_dict()
    return sorted(
        (_compare_metric(name, base_dict[name], ft_dict[name]) for name in METRIC_DIRECTIONS),
        key=lambda comparison: comparison.metric,
    )


def compare_reports(
    base_report: BenchmarkReport,
    fine_tuned_report: BenchmarkReport,
    gate_overrides: dict[str, float] | None = None,
) -> ComparisonReport:
    comparisons = compare_metrics(base_report.overall, fine_tuned_report.overall)
    gates = apply_overrides(DEFAULT_GATES, gate_overrides or {})
    return ComparisonReport(
        base=base_report,
        fine_tuned=fine_tuned_report,
        metric_comparisons=comparisons,
        regressions=[comparison.metric for comparison in comparisons if comparison.regression],
        improvements=[comparison.metric for comparison in comparisons if comparison.improvement],
        base_gates_passed=gates_passed(evaluate_gates(base_report.overall, gates)),
        fine_tuned_gates_passed=gates_passed(evaluate_gates(fine_tuned_report.overall, gates)),
    )


def format_comparison(report: ComparisonReport) -> str:
    lines = [f"{'metric':<32}{'base':>10}{'fine-tuned':>12}{'delta':>10}  verdict"]
    for comparison in report.metric_comparisons:
        verdict = "REGRESSION" if comparison.regression else ("improved" if comparison.improvement else "-")
        lines.append(
            f"{comparison.metric:<32}{comparison.base:>10.3f}{comparison.fine_tuned:>12.3f}"
            f"{comparison.delta:>+10.3f}  {verdict}"
        )
    lines.append("")
    lines.append(f"regressions: {report.regressions or 'none'}")
    lines.append(f"improvements: {report.improvements or 'none'}")
    lines.append(f"base model passes mandatory gates: {report.base_gates_passed}")
    lines.append(f"fine-tuned model passes mandatory gates: {report.fine_tuned_gates_passed}")
    return "\n".join(lines)


def comparison_to_json(report: ComparisonReport) -> dict:
    return {
        "base_overall": report.base.overall.as_dict(),
        "fine_tuned_overall": report.fine_tuned.overall.as_dict(),
        "metric_comparisons": [dataclasses.asdict(comparison) for comparison in report.metric_comparisons],
        "regressions": report.regressions,
        "improvements": report.improvements,
        "base_gates_passed": report.base_gates_passed,
        "fine_tuned_gates_passed": report.fine_tuned_gates_passed,
    }


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "usage: python -m llm_internal.eval.compare_models <base_model_config.yaml> <fine_tuned_config.yaml>",
            file=sys.stderr,
        )
        sys.exit(2)

    base_cfg = load_benchmark_eval_config(sys.argv[1])
    ft_cfg = load_benchmark_eval_config(sys.argv[2])

    base_cases = load_benchmark(base_cfg.benchmark_files)
    ft_cases = load_benchmark(ft_cfg.benchmark_files)
    if [case.id for case in base_cases] != [case.id for case in ft_cases]:
        raise ValueError("base and fine-tuned configs must point at the exact same benchmark cases")

    base_predictions = generate_for_messages(
        [case.messages for case in base_cases],
        base_cfg.backend,
        base_cfg.model_dir,
        base_cfg.max_new_tokens,
        base_cfg.model_revision,
    )
    ft_predictions = generate_for_messages(
        [case.messages for case in ft_cases],
        ft_cfg.backend,
        ft_cfg.model_dir,
        ft_cfg.max_new_tokens,
        ft_cfg.model_revision,
    )

    base_report = score_benchmark(base_cases, base_predictions, base_cfg.min_plain_chat_chars)
    ft_report = score_benchmark(ft_cases, ft_predictions, ft_cfg.min_plain_chat_chars)
    comparison = compare_reports(base_report, ft_report, ft_cfg.gate_overrides)
    print(format_comparison(comparison))

    out_path = "comparison_report.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(comparison_to_json(comparison), handle, indent=2)
    print(f"\nwrote {out_path}")
    sys.exit(0 if comparison.fine_tuned_gates_passed and not comparison.regressions else 1)


if __name__ == "__main__":
    main()
