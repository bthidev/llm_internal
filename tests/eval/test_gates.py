# tests/eval/test_gates.py
import dataclasses

import pytest

from llm_internal.eval.gates import (
    DEFAULT_GATES,
    GateSpec,
    apply_overrides,
    evaluate_gates,
    gates_passed,
)
from llm_internal.eval.metrics import BenchmarkMetrics


def _metrics(**overrides: float) -> BenchmarkMetrics:
    base = BenchmarkMetrics(
        n_cases=10,
        tool_selection_accuracy=1.0,
        tool_call_precision=1.0,
        tool_call_recall=1.0,
        false_positive_tool_rate=0.0,
        false_negative_tool_rate=0.0,
        argument_name_accuracy=1.0,
        argument_value_accuracy=1.0,
        required_argument_accuracy=1.0,
        schema_validity_rate=1.0,
        exact_tool_call_match=1.0,
        plain_chat_pass_rate=1.0,
        hallucinated_tool_name_rate=0.0,
        hallucinated_argument_rate=0.0,
        missing_required_argument_rate=0.0,
        code_correctness_rate=1.0,
    )
    return dataclasses.replace(base, **overrides)


def test_default_gates_all_pass_on_perfect_metrics():
    results = evaluate_gates(_metrics())

    assert gates_passed(results) is True
    assert all(r.passed for r in results)


def test_mandatory_gate_failure_fails_the_run():
    metrics = _metrics(tool_selection_accuracy=0.5)  # below 0.85 mandatory threshold

    results = evaluate_gates(metrics)

    assert gates_passed(results) is False
    failed = next(r for r in results if r.metric == "tool_selection_accuracy")
    assert failed.passed is False
    assert failed.mandatory is True


def test_advisory_gate_failure_does_not_fail_the_run():
    # argument_value_accuracy is advisory (mandatory=False in DEFAULT_GATES)
    metrics = _metrics(argument_value_accuracy=0.1)

    results = evaluate_gates(metrics)

    failed = next(r for r in results if r.metric == "argument_value_accuracy")
    assert failed.passed is False
    assert failed.mandatory is False
    assert gates_passed(results) is True


def test_lower_is_better_gate_fails_above_threshold():
    metrics = _metrics(false_positive_tool_rate=0.5)  # above 0.05 mandatory ceiling

    results = evaluate_gates(metrics)

    failed = next(r for r in results if r.metric == "false_positive_tool_rate")
    assert failed.passed is False
    assert gates_passed(results) is False


def test_apply_overrides_replaces_threshold_keeping_direction_and_mandatory():
    overridden = apply_overrides(DEFAULT_GATES, {"tool_selection_accuracy": 0.5})

    gate = next(g for g in overridden if g.metric == "tool_selection_accuracy")
    assert gate.threshold == 0.5
    original = next(g for g in DEFAULT_GATES if g.metric == "tool_selection_accuracy")
    assert gate.direction == original.direction
    assert gate.mandatory == original.mandatory


def test_apply_overrides_rejects_unknown_metric_name():
    with pytest.raises(ValueError, match="unknown metrics"):
        apply_overrides(DEFAULT_GATES, {"not_a_real_metric": 0.5})


def test_gate_spec_rejects_invalid_direction():
    with pytest.raises(ValueError, match="direction"):
        GateSpec("tool_selection_accuracy", "sideways", 0.5)


def test_evaluate_gates_rejects_unknown_metric_field():
    bad_gate = GateSpec("not_a_field", "higher_is_better", 0.5)

    with pytest.raises(ValueError, match="unknown metric"):
        evaluate_gates(_metrics(), gates=(bad_gate,))
