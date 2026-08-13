# tests/eval/test_metrics.py
import pytest

from llm_internal.eval.benchmark import BenchmarkCase
from llm_internal.eval.metrics import aggregate_metrics, score_benchmark, score_case

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "units": {"type": "string"}},
            "required": ["city"],
        },
    },
}
_STOCK_TOOL = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Get stock price",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
    },
}


def _tool_case(
    id: str = "t1",
    tools: list[dict] | None = None,
    expected_tool_calls: list[dict] | None = None,
    acceptable_tool_names: list[str] | None = None,
) -> BenchmarkCase:
    return BenchmarkCase(
        id=id,
        category="single_tool_selection",
        description="d",
        tools=tools if tools is not None else [_WEATHER_TOOL],
        messages=[{"role": "user", "content": "weather in Paris?"}],
        expects_tool_call=True,
        expected_tool_calls=(
            expected_tool_calls
            if expected_tool_calls is not None
            else [{"name": "get_weather", "arguments": {"city": "Paris"}}]
        ),
        acceptable_tool_names=acceptable_tool_names,
    )


def _plain_case() -> BenchmarkCase:
    return BenchmarkCase(
        id="p1",
        category="no_tool_plain_chat",
        description="d",
        tools=[],
        messages=[{"role": "user", "content": "hi"}],
        expects_tool_call=False,
    )


def test_score_case_exact_match_on_perfect_prediction():
    case = _tool_case()
    predicted = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'

    result = score_case(case, predicted)

    assert result.tool_selection_correct is True
    assert result.exact_match is True
    assert result.arg_value_correct == 1
    assert result.hallucinated_tool_names == 0
    assert result.required_present == result.required_total == 1


def test_score_case_flags_wrong_tool_name_as_selection_failure():
    case = _tool_case()
    predicted = '<tool_call>\n{"name": "get_forecast", "arguments": {"city": "Paris"}}\n</tool_call>'

    result = score_case(case, predicted)

    assert result.tool_selection_correct is False
    assert result.false_negative_tool is False  # the model did attempt a call, just the wrong one
    assert result.call_fp == 1  # the wrong-name call is an unmatched prediction
    assert result.call_fn == 1  # the expected call was never matched


def test_score_case_flags_hallucinated_tool_name_when_absent_from_declared_tools():
    case = _tool_case()
    predicted = '<tool_call>\n{"name": "totally_made_up_tool", "arguments": {}}\n</tool_call>'

    result = score_case(case, predicted)

    assert result.hallucinated_tool_names == 1


def test_score_case_flags_missing_required_argument():
    case = _tool_case()
    predicted = '<tool_call>\n{"name": "get_weather", "arguments": {}}\n</tool_call>'

    result = score_case(case, predicted)

    assert result.tool_selection_correct is True  # right tool
    assert result.required_present == 0
    assert result.required_total == 1
    assert result.exact_match is False  # arguments differ from expected


def test_score_case_flags_hallucinated_argument_not_in_schema():
    case = _tool_case()
    predicted = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", "forecast_days": 5}}\n</tool_call>'

    result = score_case(case, predicted)

    assert result.hallucinated_arg_count == 1
    assert result.arg_name_fp == 1  # "forecast_days" not in expected arguments either


def test_score_case_flags_incorrect_argument_value_distinct_from_argument_name():
    case = _tool_case()
    predicted = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "London"}}\n</tool_call>'

    result = score_case(case, predicted)

    assert result.tool_selection_correct is True
    assert result.arg_name_tp == 1  # "city" key present
    assert result.arg_value_correct == 0  # but wrong value
    assert result.exact_match is False


def test_score_case_no_tool_case_passes_on_clean_reply():
    case = _plain_case()

    result = score_case(case, "Sure, happy to help!")

    assert result.tool_selection_correct is True
    assert result.false_positive_tool is False
    assert result.plain_chat_passed is True


def test_score_case_no_tool_case_flags_unexpected_tool_call():
    case = _plain_case()
    predicted = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'

    result = score_case(case, predicted)

    assert result.tool_selection_correct is False
    assert result.false_positive_tool is True
    assert result.plain_chat_passed is None  # not evaluated once a stray call is made


def test_score_case_malformed_json_counts_as_schema_invalid_and_false_negative():
    case = _tool_case()
    predicted = "<tool_call>\nnot json\n</tool_call>"

    result = score_case(case, predicted)

    assert result.malformed_json_blocks == 1
    assert result.schema_invalid_calls == 1
    assert result.false_negative_tool is True


def test_score_case_valid_json_with_wrong_shape_counts_as_schema_invalid():
    """A <tool_call> block that parses as JSON but isn't {"name": str,
    "arguments": dict} (e.g. arguments is a list) is a distinct failure
    mode from malformed JSON -- still schema-invalid, but not counted as
    a JSON parse failure."""
    case = _tool_case()
    predicted = '<tool_call>\n{"name": "get_weather", "arguments": ["Paris"]}\n</tool_call>'

    result = score_case(case, predicted)

    assert result.malformed_json_blocks == 0
    assert result.schema_invalid_calls == 1
    assert result.schema_valid_calls == 0


def test_score_case_ambiguous_case_accepts_any_listed_tool():
    case = _tool_case(
        id="amb1",
        expected_tool_calls=[],
        acceptable_tool_names=["get_weather", "get_stock_price"],
        tools=[_WEATHER_TOOL, _STOCK_TOOL],
    )
    predicted = '<tool_call>\n{"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}\n</tool_call>'

    result = score_case(case, predicted)

    assert result.tool_selection_correct is True


def test_score_case_parallel_calls_matched_independently():
    case = _tool_case(
        id="par1",
        tools=[_WEATHER_TOOL, _STOCK_TOOL],
        expected_tool_calls=[
            {"name": "get_weather", "arguments": {"city": "Berlin"}},
            {"name": "get_stock_price", "arguments": {"symbol": "MSFT"}},
        ],
    )
    predicted = (
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Berlin"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "get_stock_price", "arguments": {"symbol": "MSFT"}}\n</tool_call>'
    )

    result = score_case(case, predicted)

    assert result.exact_match is True
    assert result.call_tp == 2
    assert result.call_fp == 0
    assert result.call_fn == 0


def test_aggregate_metrics_computes_global_and_per_category_breakdown():
    case_a = _tool_case()
    case_b = _plain_case()
    perfect = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
    results = [score_case(case_a, perfect), score_case(case_b, "All good, thanks!")]

    report = aggregate_metrics(results)

    assert report.overall.n_cases == 2
    assert report.overall.tool_selection_accuracy == 1.0
    assert set(report.by_category) == {"single_tool_selection", "no_tool_plain_chat"}
    assert report.by_category["single_tool_selection"].n_cases == 1


def test_aggregate_metrics_rejects_empty_results():
    with pytest.raises(ValueError):
        aggregate_metrics([])


def test_score_benchmark_requires_matching_lengths():
    with pytest.raises(ValueError):
        score_benchmark([_tool_case()], [])


def _code_case(id: str = "code1") -> BenchmarkCase:
    return BenchmarkCase(
        id=id,
        category="code_correctness",
        description="d",
        tools=[],
        messages=[{"role": "user", "content": "write add(a, b)"}],
        expects_tool_call=False,
        expects_code=True,
        entry_point="add",
        test_code="assert add(2, 3) == 5",
    )


def test_score_case_code_correctness_passes_on_correct_implementation():
    case = _code_case()

    result = score_case(case, "```python\ndef add(a, b):\n    return a + b\n```")

    assert result.is_code is True
    assert result.code_correctness_passed is True
    assert result.code_exec_error is None


def test_score_case_code_correctness_fails_on_incorrect_implementation():
    case = _code_case()

    result = score_case(case, "```python\ndef add(a, b):\n    return a - b\n```")

    assert result.is_code is True
    assert result.code_correctness_passed is False
    assert result.code_exec_error is not None


def test_code_case_excluded_from_tool_and_plain_chat_metrics():
    """A code case must not pollute tool_selection_accuracy/plain_chat_pass_rate
    denominators -- it's scored by a completely separate code_correctness_rate."""
    tool_case = _tool_case()
    tool_predicted = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
    code_case = _code_case()

    report = score_benchmark(
        [tool_case, code_case],
        [tool_predicted, "```python\ndef add(a, b):\n    return a - b\n```"],  # wrong code
    )

    assert report.overall.tool_selection_accuracy == 1.0  # unaffected by the failing code case
    assert report.overall.code_correctness_rate == 0.0
    assert report.by_category["code_correctness"].code_correctness_rate == 0.0


def test_argument_value_accuracy_isolated_from_tool_selection_accuracy():
    """A model that always picks the right tool but always gets one
    argument value wrong must show 1.0 selection accuracy alongside <1.0
    argument value accuracy -- neither metric should hide the other."""
    case = _tool_case()
    predicted = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Berlin"}}\n</tool_call>'

    report = score_benchmark([case], [predicted])

    assert report.overall.tool_selection_accuracy == 1.0
    assert report.overall.argument_value_accuracy == 0.0
