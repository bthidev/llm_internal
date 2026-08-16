"""CLI entrypoint for the independent tool-calling benchmark
(data/benchmark/*.jsonl). Unlike eval/run_eval.py's held-out Hermes split,
every prompt here is hand-authored and independent from the training data
(see eval/benchmark.py). Reports per-category and global metrics
(eval/metrics.py) and exits non-zero if any mandatory quality gate
(eval/gates.py) fails, so it's CI-safe.

`run_benchmark` is pure given `predictions` (no model/GPU); `main` wires it
to a real model load + generation via eval/generation.py, dispatched on
BenchmarkEvalConfig.backend. Report output is written to the configured
`report_path`; parent directories are created automatically.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

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
    """Pure: score predictions and evaluate configured quality gates.

    No model/GPU work happens here, which keeps this path unit-testable and
    allows CI to validate benchmark/gate behavior independently of hardware.
    """
    report = score_benchmark(cases, predictions, cfg.min_plain_chat_chars)
    gates = apply_overrides(DEFAULT_GATES, cfg.gate_overrides)
    return report, evaluate_gates(report.overall, gates)


def generate_benchmark_predictions(cases: list[BenchmarkCase], cfg: BenchmarkEvalConfig) -> list[str]:
    """GPU/Metal-only: generate one reply for each benchmark case.

    `case.messages` is already the complete prompt under test, unlike the
    held-out Hermes evaluation where a trailing target turn must be removed.
    """
    return generate_for_messages(
        [case.messages for case in cases],
        cfg.backend,
        cfg.model_dir,
        cfg.max_new_tokens,
        cfg.model_revision,
    )


def format_report(report: BenchmarkReport, gate_results: list[GateResult]) -> str:
    lines = ["=== overall ==="]
    for key, value in report.overall.as_dict().items():
        lines.append(f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}")
    lines.extend(["", "=== by category ==="])
    for category, metrics in report.by_category.items():
        lines.append(
            f"{category}: n={metrics.n_cases} "
            f"tool_selection_accuracy={metrics.tool_selection_accuracy:.3f} "
            f"exact_tool_call_match={metrics.exact_tool_call_match:.3f} "
            f"plain_chat_pass_rate={metrics.plain_chat_pass_rate:.3f} "
            f"code_correctness_rate={metrics.code_correctness_rate:.3f}"
        )
    lines.extend(["", "=== gates ==="])
    for gate in gate_results:
        kind = "mandatory" if gate.mandatory else "advisory"
        status = "PASS" if gate.passed else "FAIL"
        lines.append(
            f"[{status}] ({kind}) {gate.metric}={gate.value:.3f} "
            f"vs threshold {gate.threshold} ({gate.direction})"
        )
    return "\n".join(lines)


def report_to_json(report: BenchmarkReport, gate_results: list[GateResult]) -> dict:
    return {
        "overall": report.overall.as_dict(),
        "by_category": {key: value.as_dict() for key, value in report.by_category.items()},
        "gates": [dataclasses.asdict(gate) for gate in gate_results],
        "passed": gates_passed(gate_results),
    }


def _write_report(path: str, payload: dict) -> Path:
    """Write a JSON report, creating any configured parent directory."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    cfg = load_benchmark_eval_config("configs/benchmark_eval.yaml")
    cases = load_benchmark(cfg.benchmark_files)
    predictions = generate_benchmark_predictions(cases, cfg)
    report, gate_results = run_benchmark(cases, predictions, cfg)

    print(format_report(report, gate_results))
    out_path = _write_report(cfg.report_path, report_to_json(report, gate_results))
    print(f"\nwrote {out_path}")
    sys.exit(0 if gates_passed(gate_results) else 1)


if __name__ == "__main__":
    main()
