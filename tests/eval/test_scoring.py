# tests/eval/test_scoring.py
from llm_internal.eval.scoring import (
    Report,
    aggregate_results,
    parse_tool_calls,
    score_plain_chat_example,
    score_tool_call_example,
)


def test_parse_tool_calls_extracts_one_call():
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'

    calls = parse_tool_calls(text)

    assert calls == [{"name": "get_weather", "arguments": {"city": "Paris"}}]


def test_parse_tool_calls_extracts_multiple_calls():
    text = (
        '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\n'
        '<tool_call>\n{"name": "b", "arguments": {"x": 1}}\n</tool_call>'
    )

    calls = parse_tool_calls(text)

    assert [c["name"] for c in calls] == ["a", "b"]


def test_parse_tool_calls_skips_malformed_json_without_raising():
    text = "<tool_call>\nnot json\n</tool_call>"

    assert parse_tool_calls(text) == []


def test_parse_tool_calls_returns_empty_for_plain_text():
    assert parse_tool_calls("just a normal reply") == []


def test_score_tool_call_example_matches_name_and_arguments():
    expected_messages = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'},
    ]
    predicted = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'

    result = score_tool_call_example(expected_messages, predicted)

    assert result == {"correct_name": True, "correct_args": True, "structural_match": True}


def test_score_tool_call_example_flags_wrong_name():
    expected_messages = [
        {"role": "assistant", "content": '<tool_call>\n{"name": "get_weather", "arguments": {}}\n</tool_call>'},
    ]
    predicted = '<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>'

    result = score_tool_call_example(expected_messages, predicted)

    assert result["correct_name"] is False
    assert result["structural_match"] is False


def test_score_tool_call_example_flags_wrong_arguments():
    expected_messages = [
        {"role": "assistant", "content": '<tool_call>\n{"name": "f", "arguments": {"x": 1}}\n</tool_call>'},
    ]
    predicted = '<tool_call>\n{"name": "f", "arguments": {"x": 2}}\n</tool_call>'

    result = score_tool_call_example(expected_messages, predicted)

    assert result["correct_name"] is True
    assert result["correct_args"] is False
    assert result["structural_match"] is False


def test_score_plain_chat_example_passes_on_reasonable_length():
    assert score_plain_chat_example("Sure, here's the answer you asked for.", min_chars=5) is True


def test_score_plain_chat_example_fails_on_empty_or_too_short():
    assert score_plain_chat_example("", min_chars=5) is False
    assert score_plain_chat_example("ok", min_chars=5) is False


def test_aggregate_results_computes_rates_and_gate():
    results = [
        {"category": "tool_call", "structural_match": True, "plain_chat_pass": None},
        {"category": "tool_call", "structural_match": False, "plain_chat_pass": None},
        {"category": "tool_call", "structural_match": True, "plain_chat_pass": None},
        {"category": "tool_call", "structural_match": True, "plain_chat_pass": None},
        {"category": "plain_chat", "structural_match": None, "plain_chat_pass": True},
        {"category": "plain_chat", "structural_match": None, "plain_chat_pass": True},
    ]

    report = aggregate_results(results, tool_call_accuracy_threshold=0.7, plain_chat_pass_rate_threshold=0.8)

    assert isinstance(report, Report)
    assert report.tool_call_accuracy == 0.75
    assert report.plain_chat_pass_rate == 1.0
    assert report.passed is True


def test_aggregate_results_fails_gate_below_threshold():
    results = [
        {"category": "tool_call", "structural_match": False, "plain_chat_pass": None},
        {"category": "tool_call", "structural_match": True, "plain_chat_pass": None},
    ]

    report = aggregate_results(results, tool_call_accuracy_threshold=0.8, plain_chat_pass_rate_threshold=0.8)

    assert report.tool_call_accuracy == 0.5
    assert report.passed is False
