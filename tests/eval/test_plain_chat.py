# tests/eval/test_plain_chat.py
from llm_internal.eval.plain_chat import check_plain_chat_response


def test_passes_on_normal_reply():
    result = check_plain_chat_response("Sure, here's the answer you asked for.", min_chars=5)

    assert result.passed is True
    assert result.reasons == ()


def test_fails_on_empty_response():
    result = check_plain_chat_response("", min_chars=5)

    assert result.passed is False
    assert "empty_response" in result.reasons


def test_fails_on_too_short_response():
    result = check_plain_chat_response("ok", min_chars=5)

    assert result.passed is False
    assert "too_short" in result.reasons


def test_fails_on_unexpected_tool_call():
    text = 'Sure!\n<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>'

    result = check_plain_chat_response(text, min_chars=5)

    assert result.passed is False
    assert "unexpected_tool_call" in result.reasons


def test_fails_on_special_token_leakage():
    result = check_plain_chat_response("Hello there<|im_end|>", min_chars=5)

    assert result.passed is False
    assert "special_token_leakage" in result.reasons


def test_fails_on_protocol_tag_leakage():
    result = check_plain_chat_response("<tools>[]</tools> Sure, I can help.", min_chars=5)

    assert result.passed is False
    assert "protocol_tag_leakage" in result.reasons


def test_fails_on_role_prefix_leakage():
    result = check_plain_chat_response("assistant: Sure, here is the answer.", min_chars=5)

    assert result.passed is False
    assert "role_prefix_leakage" in result.reasons


def test_fails_on_degenerate_repetition():
    text = " ".join(["blah"] * 40)

    result = check_plain_chat_response(text, min_chars=5)

    assert result.passed is False
    assert "degenerate_repetition" in result.reasons


def test_passes_on_long_varied_reply_without_repetition():
    text = "This is a normal, reasonably long reply with many distinct words used throughout the sentence structure."

    result = check_plain_chat_response(text, min_chars=5)

    assert result.passed is True


def test_reports_multiple_simultaneous_failures():
    text = "assistant:<|im_end|>"

    result = check_plain_chat_response(text, min_chars=5)

    assert result.passed is False
    assert "role_prefix_leakage" in result.reasons
    assert "special_token_leakage" in result.reasons
