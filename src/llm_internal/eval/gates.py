"""Quality gates and metric direction metadata for benchmark evaluation.

`METRIC_DIRECTIONS` is the single source of truth for whether each benchmark
metric is better when higher or lower. Gate definitions and base-vs-fine-tuned
regression detection both consume it, preventing silent direction drift.
"""

from __future__ import annotations

import dataclasses

from llm_internal.eval.metrics import BenchmarkMetrics

HIGHER_IS_BETTER = "higher_is_better"
LOWER_IS_BETTER = "lower_is_better"

# Every BenchmarkMetrics field except the non-quality counter `n_cases` must
# appear exactly once here. A config smoke-test enforces that invariant.
METRIC_DIRECTIONS: dict[str, str] = {
    "tool_selection_accuracy": HIGHER_IS_BETTER,
    "tool_call_precision": HIGHER_IS_BETTER,
    "tool_call_recall": HIGHER_IS_BETTER,
    "false_positive_tool_rate": LOWER_IS_BETTER,
    "false_negative_tool_rate": LOWER_IS_BETTER,
    "argument_name_accuracy": HIGHER_IS_BETTER,
    "argument_value_accuracy": HIGHER_IS_BETTER,
    "required_argument_accuracy": HIGHER_IS_BETTER,
    "schema_validity_rate": HIGHER_IS_BETTER,
    "exact_tool_call_match": HIGHER_IS_BETTER,
    "plain_chat_pass_rate": HIGHER_IS_BETTER,
    "hallucinated_tool_name_rate": LOWER_IS_BETTER,
    "hallucinated_argument_rate": LOWER_IS_BETTER,
    "missing_required_argument_rate": LOWER_IS_BETTER,
    "code_correctness_rate": HIGHER_IS_BETTER,
}


@dataclasses.dataclass(frozen=True)
class GateSpec:
    metric: str
    direction: str
    threshold: float
    mandatory: bool = True

    def __post_init__(self) -> None:
        if self.direction not in (HIGHER_IS_BETTER, LOWER_IS_BETTER):
            raise ValueError(f"gate {self.metric!r}: invalid direction {self.direction!r}")
        expected = METRIC_DIRECTIONS.get(self.metric)
        if expected is None:
            raise ValueError(f"gate {self.metric!r}: metric has no registered direction")
        if self.direction != expected:
            raise ValueError(
                f"gate {self.metric!r}: direction {self.direction!r} disagrees with registered direction {expected!r}"
            )


def _gate(metric: str, threshold: float, *, mandatory: bool = True) -> GateSpec:
    return GateSpec(metric, METRIC_DIRECTIONS[metric], threshold, mandatory=mandatory)


DEFAULT_GATES: tuple[GateSpec, ...] = (
    _gate("tool_selection_accuracy", 0.85),
    _gate("schema_validity_rate", 0.95),
    _gate("required_argument_accuracy", 0.90),
    _gate("plain_chat_pass_rate", 0.90),
    _gate("false_positive_tool_rate", 0.05),
    _gate("false_negative_tool_rate", 0.15),
    _gate("hallucinated_tool_name_rate", 0.05),
    _gate("exact_tool_call_match", 0.60, mandatory=False),
    _gate("argument_name_accuracy", 0.85, mandatory=False),
    _gate("argument_value_accuracy", 0.70, mandatory=False),
    _gate("hallucinated_argument_rate", 0.10, mandatory=False),
    _gate("code_correctness_rate", 0.0, mandatory=False),
)


def apply_overrides(gates: tuple[GateSpec, ...], overrides: dict[str, float]) -> tuple[GateSpec, ...]:
    known = {gate.metric for gate in gates}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"gate_overrides names unknown metrics: {sorted(unknown)}")
    return tuple(
        dataclasses.replace(gate, threshold=overrides[gate.metric]) if gate.metric in overrides else gate
        for gate in gates
    )


@dataclasses.dataclass(frozen=True)
class GateResult:
    metric: str
    value: float
    threshold: float
    direction: str
    mandatory: bool
    passed: bool


def evaluate_gates(metrics: BenchmarkMetrics, gates: tuple[GateSpec, ...] = DEFAULT_GATES) -> list[GateResult]:
    values = metrics.as_dict()
    results = []
    for gate in gates:
        if gate.metric not in values:
            raise ValueError(f"unknown metric {gate.metric!r} in gate spec (not a BenchmarkMetrics field)")
        value = values[gate.metric]
        passed = value >= gate.threshold if gate.direction == HIGHER_IS_BETTER else value <= gate.threshold
        results.append(
            GateResult(
                metric=gate.metric,
                value=value,
                threshold=gate.threshold,
                direction=gate.direction,
                mandatory=gate.mandatory,
                passed=passed,
            )
        )
    return results


def gates_passed(results: list[GateResult]) -> bool:
    return all(result.passed for result in results if result.mandatory)
