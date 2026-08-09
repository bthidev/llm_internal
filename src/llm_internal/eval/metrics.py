"""Fine-grained scoring for the independent benchmark (eval/benchmark.py).
Pure functions: given a `BenchmarkCase` and its predicted text, compute a
per-case breakdown; `aggregate_metrics` rolls per-case results into global
and per-category `BenchmarkMetrics`. No model/GPU involved -- takes
already-generated prediction text, same split as eval/scoring.py.

Three questions are answered *separately*, on purpose (a task requirement):
1. tool selection      -- did the model call the right tool(s), or rightly
                           call none? (tool_selection_accuracy,
                           false_positive/negative_tool_rate,
                           tool_call_precision/recall)
2. structural validity  -- is the JSON/schema shape right (right argument
                           *names*, all required arguments present, valid
                           JSON)? (schema_validity_rate,
                           argument_name_accuracy, required_argument_accuracy,
                           hallucinated_tool_name_rate, hallucinated_argument_rate)
3. semantic correctness -- among correctly-named arguments, are the
                           *values* right? (argument_value_accuracy)

A model can ace (1) while failing (2)/(3), or vice versa; the metrics never
collapse into one score that could hide that.
"""
from __future__ import annotations

import dataclasses

from llm_internal.eval.benchmark import BenchmarkCase
from llm_internal.eval.plain_chat import check_plain_chat_response
from llm_internal.eval.scoring import parse_tool_calls_detailed


@dataclasses.dataclass(frozen=True)
class _MatchedCall:
    expected: dict
    predicted: dict


def _tool_schema_by_name(tools: list[dict]) -> dict[str, dict]:
    schemas: dict[str, dict] = {}
    for t in tools:
        fn = t.get("function", {})
        name = fn.get("name")
        if name:
            schemas[name] = fn.get("parameters", {}) or {}
    return schemas


def _match_calls(expected: list[dict], predicted: list[dict]) -> tuple[list[_MatchedCall], list[dict], list[dict]]:
    """Greedy one-to-one matching by tool name (order-preserving, so
    parallel/multiple calls are each matched independently). Returns
    (matched pairs, unmatched expected == false negatives, unmatched
    predicted == false positives)."""
    remaining_predicted = list(predicted)
    matched: list[_MatchedCall] = []
    unmatched_expected: list[dict] = []
    for exp in expected:
        idx = next((i for i, p in enumerate(remaining_predicted) if p.get("name") == exp.get("name")), None)
        if idx is None:
            unmatched_expected.append(exp)
        else:
            matched.append(_MatchedCall(expected=exp, predicted=remaining_predicted.pop(idx)))
    return matched, unmatched_expected, remaining_predicted


@dataclasses.dataclass(frozen=True)
class CaseResult:
    id: str
    category: str
    expects_tool_call: bool

    # protocol-level (every predicted <tool_call> block, matched or not)
    predicted_call_count: int
    malformed_json_blocks: int
    schema_valid_calls: int
    schema_invalid_calls: int
    hallucinated_tool_names: int  # predicted calls naming a tool absent from case.tools

    # case-level tool-selection verdicts
    tool_selection_correct: bool
    exact_match: bool
    false_positive_tool: bool  # called a tool when none was expected
    false_negative_tool: bool  # expected a call, model made none

    # per-call micro counts (name-only matching), for precision/recall
    call_tp: int = 0
    call_fp: int = 0
    call_fn: int = 0

    # argument-level counts, summed over matched (expected, predicted) call pairs
    arg_name_tp: int = 0
    arg_name_fp: int = 0
    arg_name_fn: int = 0
    arg_value_correct: int = 0  # of arg_name_tp, how many also have the right value
    required_total: int = 0
    required_present: int = 0
    hallucinated_arg_count: int = 0
    predicted_arg_count: int = 0

    plain_chat_passed: bool | None = None  # only set when not expects_tool_call and no stray call
    plain_chat_reasons: tuple[str, ...] = ()


def score_case(case: BenchmarkCase, predicted_text: str, min_plain_chat_chars: int = 5) -> CaseResult:
    predicted_calls_raw, malformed = parse_tool_calls_detailed(predicted_text)

    # Structural validity: a parsed block counts as schema-valid only if it
    # has a string "name" and a dict "arguments" -- the minimum shape any
    # downstream tool executor requires.
    predicted_calls: list[dict] = []
    schema_valid = 0
    schema_invalid = malformed
    for call in predicted_calls_raw:
        name = call.get("name")
        args = call.get("arguments")
        if isinstance(name, str) and isinstance(args, dict):
            schema_valid += 1
            predicted_calls.append(call)
        else:
            schema_invalid += 1

    known_tool_names = {t.get("function", {}).get("name") for t in case.tools}
    hallucinated_tool_names = sum(1 for c in predicted_calls if c["name"] not in known_tool_names)

    if not case.expects_tool_call:
        false_positive = len(predicted_calls_raw) > 0
        plain_chat = None if false_positive else check_plain_chat_response(predicted_text, min_plain_chat_chars)
        return CaseResult(
            id=case.id, category=case.category, expects_tool_call=False,
            predicted_call_count=len(predicted_calls_raw), malformed_json_blocks=malformed,
            schema_valid_calls=schema_valid, schema_invalid_calls=schema_invalid,
            hallucinated_tool_names=hallucinated_tool_names,
            tool_selection_correct=not false_positive, exact_match=not false_positive,
            false_positive_tool=false_positive, false_negative_tool=False,
            call_tp=0, call_fp=len(predicted_calls), call_fn=0,
            plain_chat_passed=(plain_chat.passed if plain_chat else None),
            plain_chat_reasons=(plain_chat.reasons if plain_chat else ()),
        )

    expected_names = (
        set(case.acceptable_tool_names) if case.acceptable_tool_names
        else {c["name"] for c in case.expected_tool_calls}
    )
    predicted_names = {c["name"] for c in predicted_calls}
    false_negative = len(predicted_calls) == 0

    if case.acceptable_tool_names:
        # Ambiguous case: any single call into the acceptable set is a
        # correct selection; there's no single ground truth for arguments,
        # so argument-level metrics are skipped for these calls.
        tool_selection_correct = len(predicted_calls) >= 1 and predicted_names <= expected_names
        exact_match = tool_selection_correct and len(predicted_calls) == 1
        call_tp = len(predicted_calls) if tool_selection_correct else 0
        call_fp = 0 if tool_selection_correct else len(predicted_calls)
        call_fn = 0 if predicted_calls else 1
        matched: list[_MatchedCall] = []
    else:
        tool_selection_correct = predicted_names == expected_names
        matched, unmatched_expected, unmatched_predicted = _match_calls(case.expected_tool_calls, predicted_calls)
        exact_match = (
            not unmatched_expected and not unmatched_predicted
            and all(m.expected.get("arguments") == m.predicted.get("arguments") for m in matched)
        )
        call_tp, call_fp, call_fn = len(matched), len(unmatched_predicted), len(unmatched_expected)

    schemas = _tool_schema_by_name(case.tools)
    arg_name_tp = arg_name_fp = arg_name_fn = 0
    arg_value_correct = 0
    required_total = required_present = 0
    hallucinated_arg_count = 0
    predicted_arg_count = 0
    for m in matched:
        exp_args = m.expected.get("arguments") or {}
        pred_args = m.predicted.get("arguments") or {}
        exp_keys, pred_keys = set(exp_args), set(pred_args)
        predicted_arg_count += len(pred_keys)

        tp_keys = exp_keys & pred_keys
        arg_name_tp += len(tp_keys)
        arg_name_fp += len(pred_keys - exp_keys)
        arg_name_fn += len(exp_keys - pred_keys)
        arg_value_correct += sum(1 for k in tp_keys if pred_args[k] == exp_args[k])

        schema = schemas.get(m.expected.get("name", ""), {})
        required = set(schema.get("required", []))
        properties = set(schema.get("properties", {}))
        required_total += len(required)
        required_present += len(required & pred_keys)
        if properties:
            hallucinated_arg_count += len(pred_keys - properties)

    return CaseResult(
        id=case.id, category=case.category, expects_tool_call=True,
        predicted_call_count=len(predicted_calls_raw), malformed_json_blocks=malformed,
        schema_valid_calls=schema_valid, schema_invalid_calls=schema_invalid,
        hallucinated_tool_names=hallucinated_tool_names,
        tool_selection_correct=tool_selection_correct, exact_match=exact_match,
        false_positive_tool=False, false_negative_tool=false_negative,
        call_tp=call_tp, call_fp=call_fp, call_fn=call_fn,
        arg_name_tp=arg_name_tp, arg_name_fp=arg_name_fp, arg_name_fn=arg_name_fn,
        arg_value_correct=arg_value_correct,
        required_total=required_total, required_present=required_present,
        hallucinated_arg_count=hallucinated_arg_count, predicted_arg_count=predicted_arg_count,
    )


def _safe_div(numerator: float, denominator: float, default: float = 1.0) -> float:
    return numerator / denominator if denominator else default


@dataclasses.dataclass(frozen=True)
class BenchmarkMetrics:
    n_cases: int
    tool_selection_accuracy: float
    tool_call_precision: float
    tool_call_recall: float
    false_positive_tool_rate: float
    false_negative_tool_rate: float
    argument_name_accuracy: float
    argument_value_accuracy: float
    required_argument_accuracy: float
    schema_validity_rate: float
    exact_tool_call_match: float
    plain_chat_pass_rate: float
    hallucinated_tool_name_rate: float
    hallucinated_argument_rate: float
    missing_required_argument_rate: float

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _compute_metrics(results: list[CaseResult]) -> BenchmarkMetrics:
    tool_call_cases = [r for r in results if r.expects_tool_call]
    no_call_cases = [r for r in results if not r.expects_tool_call]

    call_tp = sum(r.call_tp for r in results)
    call_fp = sum(r.call_fp for r in results)
    call_fn = sum(r.call_fn for r in results)

    arg_name_tp = sum(r.arg_name_tp for r in results)
    arg_name_fp = sum(r.arg_name_fp for r in results)
    arg_name_fn = sum(r.arg_name_fn for r in results)
    arg_value_correct = sum(r.arg_value_correct for r in results)
    required_total = sum(r.required_total for r in results)
    required_present = sum(r.required_present for r in results)
    hallucinated_arg_count = sum(r.hallucinated_arg_count for r in results)
    predicted_arg_count = sum(r.predicted_arg_count for r in results)

    schema_valid = sum(r.schema_valid_calls for r in results)
    schema_invalid = sum(r.schema_invalid_calls for r in results)
    hallucinated_tool_names = sum(r.hallucinated_tool_names for r in results)
    total_predicted_calls = sum(r.predicted_call_count for r in results)

    plain_chat_results = [r.plain_chat_passed for r in no_call_cases if r.plain_chat_passed is not None]

    return BenchmarkMetrics(
        n_cases=len(results),
        tool_selection_accuracy=_safe_div(sum(r.tool_selection_correct for r in results), len(results)),
        tool_call_precision=_safe_div(call_tp, call_tp + call_fp),
        tool_call_recall=_safe_div(call_tp, call_tp + call_fn),
        false_positive_tool_rate=_safe_div(sum(r.false_positive_tool for r in no_call_cases), len(no_call_cases), 0.0),
        false_negative_tool_rate=_safe_div(sum(r.false_negative_tool for r in tool_call_cases), len(tool_call_cases), 0.0),
        argument_name_accuracy=_safe_div(2 * arg_name_tp, 2 * arg_name_tp + arg_name_fp + arg_name_fn),
        argument_value_accuracy=_safe_div(arg_value_correct, arg_name_tp),
        required_argument_accuracy=_safe_div(required_present, required_total),
        schema_validity_rate=_safe_div(schema_valid, schema_valid + schema_invalid),
        exact_tool_call_match=_safe_div(sum(r.exact_match for r in results), len(results)),
        plain_chat_pass_rate=_safe_div(sum(bool(p) for p in plain_chat_results), len(plain_chat_results)),
        hallucinated_tool_name_rate=_safe_div(hallucinated_tool_names, total_predicted_calls, 0.0),
        hallucinated_argument_rate=_safe_div(hallucinated_arg_count, predicted_arg_count, 0.0),
        missing_required_argument_rate=_safe_div(required_total - required_present, required_total, 0.0),
    )


@dataclasses.dataclass(frozen=True)
class BenchmarkReport:
    overall: BenchmarkMetrics
    by_category: dict[str, BenchmarkMetrics]
    case_results: list[CaseResult]


def aggregate_metrics(results: list[CaseResult]) -> BenchmarkReport:
    if not results:
        raise ValueError("aggregate_metrics requires at least one case result")
    by_category: dict[str, list[CaseResult]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)
    return BenchmarkReport(
        overall=_compute_metrics(results),
        by_category={cat: _compute_metrics(rs) for cat, rs in sorted(by_category.items())},
        case_results=results,
    )


def score_benchmark(
    cases: list[BenchmarkCase], predictions: list[str], min_plain_chat_chars: int = 5
) -> BenchmarkReport:
    if len(cases) != len(predictions):
        raise ValueError(f"cases ({len(cases)}) and predictions ({len(predictions)}) length mismatch")
    results = [score_case(c, p, min_plain_chat_chars) for c, p in zip(cases, predictions, strict=True)]
    return aggregate_metrics(results)
