"""Quality gates for the independent benchmark report (eval/metrics.py).

Each gate names a `BenchmarkMetrics` field, a comparison direction, and a
threshold. `mandatory` gates must pass for the run to succeed -- their
failure is what flips the CLI's exit code, so `run_benchmark`/`compare_models`
can gate CI. Non-mandatory ("advisory") gates are reported with the same
pass/fail verdict but never block: they cover metrics that are either
inherently noisy at this benchmark's size (`exact_tool_call_match`,
`argument_value_accuracy` -- free-text/date values can be phrased
correctly in more than one way) or are early-warning signals rather than
hard requirements (`hallucinated_argument_rate`).

Defaults (rationale):
- tool_selection_accuracy >= 0.85 -- the fine-tune's core job: pick the
  right tool, or rightly pick none.
- schema_validity_rate >= 0.95 -- malformed tool-call JSON breaks any
  downstream executor; this should be near-perfect regardless of semantic
  correctness.
- required_argument_accuracy >= 0.90 -- a call missing a required argument
  is unusable even if the tool selection was correct.
- plain_chat_pass_rate >= 0.90 -- fine-tuning for tool use must not wreck
  ordinary conversation.
- false_positive_tool_rate <= 0.05 -- over-triggering (calling a tool when
  none should be called) has real side effects and must stay rare.
- false_negative_tool_rate <= 0.15 -- missing an obligatory call is
  costly, but harder to fully eliminate than over-triggering (looser bar).
- hallucinated_tool_name_rate <= 0.05 -- inventing tools that don't exist
  is a severe, easily-detected failure mode.

Override any threshold via `BenchmarkEvalConfig.gate_overrides` in
`configs/benchmark_eval.yaml` (a flat `{metric_name: threshold}` map);
direction and mandatory-ness are fixed per metric to keep gate semantics
unambiguous.
"""
from __future__ import annotations

import dataclasses

from llm_internal.eval.metrics import BenchmarkMetrics

HIGHER_IS_BETTER = "higher_is_better"
LOWER_IS_BETTER = "lower_is_better"


@dataclasses.dataclass(frozen=True)
class GateSpec:
    metric: str
    direction: str
    threshold: float
    mandatory: bool = True

    def __post_init__(self) -> None:
        if self.direction not in (HIGHER_IS_BETTER, LOWER_IS_BETTER):
            raise ValueError(f"gate {self.metric!r}: direction must be {HIGHER_IS_BETTER!r} or {LOWER_IS_BETTER!r}")


DEFAULT_GATES: tuple[GateSpec, ...] = (
    GateSpec("tool_selection_accuracy", HIGHER_IS_BETTER, 0.85, mandatory=True),
    GateSpec("schema_validity_rate", HIGHER_IS_BETTER, 0.95, mandatory=True),
    GateSpec("required_argument_accuracy", HIGHER_IS_BETTER, 0.90, mandatory=True),
    GateSpec("plain_chat_pass_rate", HIGHER_IS_BETTER, 0.90, mandatory=True),
    GateSpec("false_positive_tool_rate", LOWER_IS_BETTER, 0.05, mandatory=True),
    GateSpec("false_negative_tool_rate", LOWER_IS_BETTER, 0.15, mandatory=True),
    GateSpec("hallucinated_tool_name_rate", LOWER_IS_BETTER, 0.05, mandatory=True),
    GateSpec("exact_tool_call_match", HIGHER_IS_BETTER, 0.60, mandatory=False),
    GateSpec("argument_name_accuracy", HIGHER_IS_BETTER, 0.85, mandatory=False),
    GateSpec("argument_value_accuracy", HIGHER_IS_BETTER, 0.70, mandatory=False),
    GateSpec("hallucinated_argument_rate", LOWER_IS_BETTER, 0.10, mandatory=False),
)


def apply_overrides(gates: tuple[GateSpec, ...], overrides: dict[str, float]) -> tuple[GateSpec, ...]:
    """Replace each named gate's threshold with `overrides[metric]`,
    keeping its direction/mandatory-ness. Raises on an override naming a
    metric with no default gate."""
    known = {g.metric for g in gates}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"gate_overrides names unknown metrics: {sorted(unknown)}")
    return tuple(
        dataclasses.replace(g, threshold=overrides[g.metric]) if g.metric in overrides else g
        for g in gates
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
        results.append(GateResult(
            metric=gate.metric, value=value, threshold=gate.threshold,
            direction=gate.direction, mandatory=gate.mandatory, passed=passed,
        ))
    return results


def gates_passed(results: list[GateResult]) -> bool:
    """True iff every *mandatory* gate passed; advisory gate failures are
    reported but never affect this verdict."""
    return all(r.passed for r in results if r.mandatory)
