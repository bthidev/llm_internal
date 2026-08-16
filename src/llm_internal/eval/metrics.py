"""Fine-grained scoring for the independent benchmark."""

from __future__ import annotations

import dataclasses
from collections import Counter

from llm_internal.eval.benchmark import BenchmarkCase
from llm_internal.eval.code_exec import run_code_case
from llm_internal.eval.plain_chat import check_plain_chat_response
from llm_internal.eval.scoring import parse_tool_calls_detailed


@dataclasses.dataclass(frozen=True)
class _MatchedCall:
    expected: dict
    predicted: dict


def _tool_schema_by_name(tools: list[dict]) -> dict[str, dict]:
    schemas: dict[str, dict] = {}
    for tool in tools:
        fn = tool.get("function", {})
        name = fn.get("name")
        if name:
            schemas[name] = fn.get("parameters", {}) or {}
    return schemas


def _match_calls(expected: list[dict], predicted: list[dict]) -> tuple[list[_MatchedCall], list[dict], list[dict]]:
    remaining_predicted = list(predicted)
    matched: list[_MatchedCall] = []
    unmatched_expected: list[dict] = []
    for exp in expected:
        idx = next((i for i, pred in enumerate(remaining_predicted) if pred.get("name") == exp.get("name")), None)
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
    predicted_call_count: int
    malformed_json_blocks: int
    schema_valid_calls: int
    schema_invalid_calls: int
    hallucinated_tool_names: int
    tool_selection_correct: bool
    exact_match: bool
    false_positive_tool: bool
    false_negative_tool: bool
    call_tp: int = 0
    call_fp: int = 0
    call_fn: int = 0
    arg_name_tp: int = 0
    arg_name_fp: int = 0
    arg_name_fn: int = 0
    arg_value_correct: int = 0
    required_total: int = 0
    required_present: int = 0
    hallucinated_arg_count: int = 0
    predicted_arg_count: int = 0
    plain_chat_passed: bool | None = None
    plain_chat_reasons: tuple[str, ...] = ()
    is_code: bool = False
    code_correctness_passed: bool | None = None
    code_exec_error: str | None = None


def score_case(case: BenchmarkCase, predicted_text: str, min_plain_chat_chars: int = 5) -> CaseResult:
    if case.expects_code:
        exec_result = run_code_case(predicted_text, case.entry_point, case.test_code, case.timeout_s)
        return CaseResult(
            id=case.id,
            category=case.category,
            expects_tool_call=False,
            predicted_call_count=0,
            malformed_json_blocks=0,
            schema_valid_calls=0,
            schema_invalid_calls=0,
            hallucinated_tool_names=0,
            tool_selection_correct=True,
            exact_match=exec_result.passed,
            false_positive_tool=False,
            false_negative_tool=False,
            is_code=True,
            code_correctness_passed=exec_result.passed,
            code_exec_error=exec_result.error,
        )

    predicted_calls_raw, malformed = parse_tool_calls_detailed(predicted_text)
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

    known_tool_names = {tool.get("function", {}).get("name") for tool in case.tools}
    hallucinated_tool_names = sum(1 for call in predicted_calls if call["name"] not in known_tool_names)

    if not case.expects_tool_call:
        false_positive = len(predicted_calls_raw) > 0
        plain_chat = None if false_positive else check_plain_chat_response(predicted_text, min_plain_chat_chars)
        return CaseResult(
            id=case.id,
            category=case.category,
            expects_tool_call=False,
            predicted_call_count=len(predicted_calls_raw),
            malformed_json_blocks=malformed,
            schema_valid_calls=schema_valid,
            schema_invalid_calls=schema_invalid,
            hallucinated_tool_names=hallucinated_tool_names,
            tool_selection_correct=not false_positive,
            exact_match=not false_positive,
            false_positive_tool=false_positive,
            false_negative_tool=False,
            call_fp=len(predicted_calls),
            plain_chat_passed=(plain_chat.passed if plain_chat else None),
            plain_chat_reasons=(plain_chat.reasons if plain_chat else ()),
        )

    predicted_names = [call["name"] for call in predicted_calls]
    false_negative = len(predicted_calls) == 0

    if case.acceptable_tool_names:
        acceptable_names = set(case.acceptable_tool_names)
        tool_selection_correct = len(predicted_calls) >= 1 and set(predicted_names) <= acceptable_names
        exact_match = tool_selection_correct and len(predicted_calls) == 1
        call_tp = len(predicted_calls) if tool_selection_correct else 0
        call_fp = 0 if tool_selection_correct else len(predicted_calls)
        call_fn = 0 if predicted_calls else 1
        matched: list[_MatchedCall] = []
    else:
        expected_names = [call["name"] for call in case.expected_tool_calls]
        tool_selection_correct = Counter(predicted_names) == Counter(expected_names)
        matched, unmatched_expected, unmatched_predicted = _match_calls(case.expected_tool_calls, predicted_calls)
        exact_match = (
            not unmatched_expected
            and not unmatched_predicted
            and all(match.expected.get("arguments") == match.predicted.get("arguments") for match in matched)
        )
        call_tp, call_fp, call_fn = len(matched), len(unmatched_predicted), len(unmatched_expected)

    schemas = _tool_schema_by_name(case.tools)
    arg_name_tp = arg_name_fp = arg_name_fn = 0
    arg_value_correct = 0
    required_total = required_present = 0
    hallucinated_arg_count = 0
    predicted_arg_count = 0
    for match in matched:
        exp_args = match.expected.get("arguments") or {}
        pred_args = match.predicted.get("arguments") or {}
        exp_keys, pred_keys = set(exp_args), set(pred_args)
        predicted_arg_count += len(pred_keys)
        tp_keys = exp_keys & pred_keys
        arg_name_tp += len(tp_keys)
        arg_name_fp += len(pred_keys - exp_keys)
        arg_name_fn += len(exp_keys - pred_keys)
        arg_value_correct += sum(1 for key in tp_keys if pred_args[key] == exp_args[key])

        schema = schemas.get(match.expected.get("name", ""), {})
        required = set(schema.get("required", []))
        properties = set(schema.get("properties", {}))
        required_total += len(required)
        required_present += len(required & pred_keys)
        if properties:
            hallucinated_arg_count += len(pred_keys - properties)

    return CaseResult(
        id=case.id,
        category=case.category,
        expects_tool_call=True,
        predicted_call_count=len(predicted_calls_raw),
        malformed_json_blocks=malformed,
        schema_valid_calls=schema_valid,
        schema_invalid_calls=schema_invalid,
        hallucinated_tool_names=hallucinated_tool_names,
        tool_selection_correct=tool_selection_correct,
        exact_match=exact_match,
        false_positive_tool=False,
        false_negative_tool=false_negative,
        call_tp=call_tp,
        call_fp=call_fp,
        call_fn=call_fn,
        arg_name_tp=arg_name_tp,
        arg_name_fp=arg_name_fp,
        arg_name_fn=arg_name_fn,
        arg_value_correct=arg_value_correct,
        required_total=required_total,
        required_present=required_present,
        hallucinated_arg_count=hallucinated_arg_count,
        predicted_arg_count=predicted_arg_count,
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
    code_correctness_rate: float

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _compute_metrics(results: list[CaseResult]) -> BenchmarkMetrics:
    tc = [result for result in results if not result.is_code]
    code_results = [result for result in results if result.is_code]
    tool_call_cases = [result for result in tc if result.expects_tool_call]
    no_call_cases = [result for result in tc if not result.expects_tool_call]

    call_tp = sum(result.call_tp for result in tc)
    call_fp = sum(result.call_fp for result in tc)
    call_fn = sum(result.call_fn for result in tc)
    arg_name_tp = sum(result.arg_name_tp for result in tc)
    arg_name_fp = sum(result.arg_name_fp for result in tc)
    arg_name_fn = sum(result.arg_name_fn for result in tc)
    arg_value_correct = sum(result.arg_value_correct for result in tc)
    required_total = sum(result.required_total for result in tc)
    required_present = sum(result.required_present for result in tc)
    hallucinated_arg_count = sum(result.hallucinated_arg_count for result in tc)
    predicted_arg_count = sum(result.predicted_arg_count for result in tc)
    schema_valid = sum(result.schema_valid_calls for result in tc)
    schema_invalid = sum(result.schema_invalid_calls for result in tc)
    hallucinated_tool_names = sum(result.hallucinated_tool_names for result in tc)
    total_predicted_calls = sum(result.predicted_call_count for result in tc)
    plain_chat_results = [r.plain_chat_passed for r in no_call_cases if r.plain_chat_passed is not None]
    code_correctness_results = [r.code_correctness_passed for r in code_results if r.code_correctness_passed is not None]

    return BenchmarkMetrics(
        n_cases=len(results),
        tool_selection_accuracy=_safe_div(sum(r.tool_selection_correct for r in tc), len(tc)),
        tool_call_precision=_safe_div(call_tp, call_tp + call_fp),
        tool_call_recall=_safe_div(call_tp, call_tp + call_fn),
        false_positive_tool_rate=_safe_div(sum(r.false_positive_tool for r in no_call_cases), len(no_call_cases), 0.0),
        false_negative_tool_rate=_safe_div(sum(r.false_negative_tool for r in tool_call_cases), len(tool_call_cases), 0.0),
        argument_name_accuracy=_safe_div(2 * arg_name_tp, 2 * arg_name_tp + arg_name_fp + arg_name_fn),
        argument_value_accuracy=_safe_div(arg_value_correct, arg_name_tp),
        required_argument_accuracy=_safe_div(required_present, required_total),
        schema_validity_rate=_safe_div(schema_valid, schema_valid + schema_invalid),
        exact_tool_call_match=_safe_div(sum(r.exact_match for r in tc), len(tc)),
        plain_chat_pass_rate=_safe_div(sum(bool(value) for value in plain_chat_results), len(plain_chat_results)),
        hallucinated_tool_name_rate=_safe_div(hallucinated_tool_names, total_predicted_calls, 0.0),
        hallucinated_argument_rate=_safe_div(hallucinated_arg_count, predicted_arg_count, 0.0),
        missing_required_argument_rate=_safe_div(required_total - required_present, required_total, 0.0),
        code_correctness_rate=_safe_div(sum(bool(value) for value in code_correctness_results), len(code_correctness_results), 1.0),
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
    for result in results:
        by_category.setdefault(result.category, []).append(result)
    return BenchmarkReport(
        overall=_compute_metrics(results),
        by_category={category: _compute_metrics(category_results) for category, category_results in sorted(by_category.items())},
        case_results=results,
    )


def score_benchmark(cases: list[BenchmarkCase], predictions: list[str], min_plain_chat_chars: int = 5) -> BenchmarkReport:
    if len(cases) != len(predictions):
        raise ValueError(f"cases ({len(cases)}) and predictions ({len(predictions)}) length mismatch")
    results = [score_case(case, prediction, min_plain_chat_chars) for case, prediction in zip(cases, predictions, strict=True)]
    return aggregate_metrics(results)
