"""CLI entrypoint for the independent tool-calling benchmark
(data/benchmark/*.jsonl). Unlike eval/run_eval.py's held-out Hermes split,
every prompt here is hand-authored and independent from the training data
(see eval/benchmark.py). Reports per-category and global metrics
(eval/metrics.py) and exits non-zero if any mandatory quality gate
(eval/gates.py) fails, so it's CI-safe.

`run_benchmark` is pure given `predictions` (no model/GPU); `main` wires it
to a real model load + generation via eval/generation.py, dispatched on
BenchmarkEvalConfig.backend."""

from __future__ import annotations

import dataclasses
import json
import sys

from llm_internal.eval.benchmark import BenchmarkCase, load_benchmark
from llm_internal.eval.config import BenchmarkEvalConfig, load_benchmark_eval_config
from llm_internal.eval.gates import DEFAULT_GATES, GateResult, apply_overrides, evaluate_gates, gates_passed
from llm_internal.eval.generation import generate_for_messages
from llm_internal.eval.metrics import BenchmarkReport, score_benchmark


def run_benchmark(
    cases: list[BenchmarkCase],
    predictions: list[str],
    cfg: BenchmarkEvalConfig,
) -> tuple[BenchmarkReport, list[GateResult]]:
    """Pure: scores `predictions` against `cases` and evaluates the
    configured quality gates. No model/GPU involved."""
    report = score_benchmark(cases, predictions, cfg.min_plain_chat_chars)
    gates = apply_overrides(DEFAULT_GATES, cfg.gate_overrides)
    gate_results = evaluate_gates(report.overall, gates)
    return report, gate_results


def generate_benchmark_predictions(cases: list[BenchmarkCase], cfg: BenchmarkEvalConfig) -> list[str]:
    """GPU/Metal-only: loads cfg.model_dir and generates a reply for each
    case's `messages` (already the full prompt -- no trailing turn to
    drop, unlike the Hermes split)."""
    return generate_for_messages(
        [c.messages for c in cases],
        cfg.backend,
        cfg.model_dir,
        cfg.max_new_tokens,
        cfg.model_revision,
    )


def format_report(report: BenchmarkReport, gate_results: list[GateResult]) -> str:
    lines = ["=== overall ==="]
    for k, v in report.overall.as_dict().items():
        lines.append(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
    lines.append("")
    lines.append("=== by category ===")
    for category, metrics in report.by_category.items():
        lines.append(
            f"{category}: n={metrics.n_cases} "
            f"tool_selection_accuracy={metrics.tool_selection_accuracy:.3f} "
            f"exact_tool_call_match={metrics.exact_tool_call_match:.3f} "
            f"plain_chat_pass_rate={metrics.plain_chat_pass_rate:.3f} "
            f"code_correctness_rate={metrics.code_correctness_rate:.3f}"
        )
    lines.append("")
    lines.append("=== gates ===")
    for g in gate_results:
        kind = "mandatory" if g.mandatory else "advisory"
        status = "PASS" if g.passed else "FAIL"
        lines.append(f"[{status}] ({kind}) {g.metric}={g.value:.3f} vs threshold {g.threshold} ({g.direction})")
    return "\n".join(lines)


def report_to_json(report: BenchmarkReport, gate_results: list[GateResult]) -> dict:
    return {
        "overall": report.overall.as_dict(),
        "by_category": {k: v.as_dict() for k, v in report.by_category.items()},
        "gates": [dataclasses.asdict(g) for g in gate_results],
        "passed": gates_passed(gate_results),
    }


def main() -> None:
    cfg = load_benchmark_eval_config("configs/benchmark_eval.yaml")
    cases = load_benchmark(cfg.benchmark_files)

    predictions = generate_benchmark_predictions(cases, cfg)
    report, gate_results = run_benchmark(cases, predictions, cfg)

    print(format_report(report, gate_results))

    out_path = "benchmark_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_to_json(report, gate_results), f, indent=2)
    print(f"\nwrote {out_path}")

    sys.exit(0 if gates_passed(gate_results) else 1)


if __name__ == "__main__":
    main()
