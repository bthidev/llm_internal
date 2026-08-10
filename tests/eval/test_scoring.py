# tests/eval/test_scoring.py
from llm_internal.eval.scoring import (
    Report,
    aggregate_results,
    parse_tool_calls,
    prompt_messages_for_example,
    score_plain_chat_example,
    score_tool_call_example,
    summarize_tool_call_failures,
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



def _example(messages, category="tool_call"):
    return {"id": "x", "messages": messages, "category": category}


def test_summarize_tool_call_failures_counts_missing_call():
    examples = [_example([
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'},
    ])]
    predictions = ["Sure, let me help with that."]

    summary = summarize_tool_call_failures(examples, predictions)

    assert "no_tool_call_block=1" in summary
    assert "malformed_json=0" in summary
    assert "wrong_name=0" in summary
    assert "wrong_args=0" in summary


def test_summarize_tool_call_failures_counts_malformed_wrong_name_and_args():
    expected_call = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
    examples = [
        _example([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": expected_call},
        ]),
        _example([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": expected_call},
        ]),
        _example([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": expected_call},
        ]),
        _example([{"role": "user", "content": "hi"}], category="plain_chat"),
    ]
    predictions = [
        "<tool_call>\nnot json\n</tool_call>",
        '<tool_call>\n{"name": "get_time", "arguments": {"city": "Paris"}}\n</tool_call>',
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "London"}}\n</tool_call>',
        "hello there",
    ]

    summary = summarize_tool_call_failures(examples, predictions)

    assert "no_tool_call_block=0" in summary
    assert "malformed_json=1" in summary
    assert "wrong_name=1" in summary
    assert "wrong_args=1" in summary


def test_summarize_tool_call_failures_truncates_samples():
    expected_call = '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>'
    examples = [
        _example([{"role": "user", "content": "?"}, {"role": "assistant", "content": expected_call}])
        for _ in range(10)
    ]
    predictions = ["no call here"] * 10

    summary = summarize_tool_call_failures(examples, predictions, max_samples=2)

    assert summary.count("expected=") == 2
    assert "no_tool_call_block=10" in summary


def test_prompt_messages_for_example_drops_only_last_turn_for_plain_chat():
    example = _example(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello there"},
        ],
        category="plain_chat",
    )

    assert prompt_messages_for_example(example) == [{"role": "user", "content": "hi"}]


def test_prompt_messages_for_example_truncates_before_tool_call_when_last_turn():
    tool_call_msg = {
        "role": "assistant",
        "content": '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>',
    }
    example = _example([{"role": "user", "content": "?"}, tool_call_msg])

    assert prompt_messages_for_example(example) == [{"role": "user", "content": "?"}]


def test_prompt_messages_for_example_truncates_before_tool_call_when_conversation_continues():
    # Regression: raw hermes-function-calling-v1 conversations often continue
    # past the tool call (tool response, then a final NL summary). Dropping
    # only the last message would prompt the model to continue *after* the
    # tool call -- producing exactly the plain-text summary it's supposed
    # to, while scoring still compares that against the tool call itself,
    # guaranteeing a mismatch. The prompt must stop right before the same
    # turn _last_expected_tool_call / score_tool_call_example score against.
    user_msg = {"role": "user", "content": "book it"}
    tool_call_msg = {
        "role": "assistant",
        "content": '<tool_call>\n{"name": "book", "arguments": {"id": 1}}\n</tool_call>',
    }
    tool_response_msg = {"role": "tool", "content": '<tool_response>\n{"ok": true}\n</tool_response>'}
    summary_msg = {"role": "assistant", "content": "Done, it's booked."}
    example = _example([user_msg, tool_call_msg, tool_response_msg, summary_msg])

    assert prompt_messages_for_example(example) == [user_msg]


def test_prompt_messages_for_example_falls_back_when_no_tool_call_present():
    example = _example([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])

    assert prompt_messages_for_example(example) == [{"role": "user", "content": "hi"}]