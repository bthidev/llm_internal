# tests/data/test_transform.py
import pytest

from llm_internal.data.transform import dedupe_examples, filter_malformed_tool_calls, format_example, stratified_split


def _raw(conversations, ex_id="abc"):
    return {"id": ex_id, "conversations": conversations}


def test_format_example_maps_roles_and_flags_tool_call():
    raw = _raw([
        {"from": "system", "value": "You are a function calling AI model.\n<tools>[]</tools>"},
        {"from": "human", "value": "What's the weather?"},
        {"from": "gpt", "value": '<tool_call>\n{"name": "get_weather", "arguments": {}}\n</tool_call>'},
        {"from": "tool", "value": "<tool_response>\n{\"temp\": 20}\n</tool_response>"},
        {"from": "gpt", "value": "It's 20 degrees."},
    ])

    result = format_example(raw)

    assert result["id"] == "abc"
    assert [m["role"] for m in result["messages"]] == ["system", "user", "assistant", "tool", "assistant"]
    assert result["messages"][1]["content"] == "What's the weather?"
    assert result["category"] == "tool_call"


def test_format_example_plain_chat_has_no_tool_call_tag():
    raw = _raw([
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "human", "value": "Hi"},
        {"from": "gpt", "value": "Hello! How can I help?"},
    ])

    result = format_example(raw)

    assert result["category"] == "plain_chat"


def test_format_example_rejects_unknown_role():
    raw = _raw([{"from": "narrator", "value": "..."}])

    with pytest.raises(ValueError, match="narrator"):
        format_example(raw)


def test_stratified_split_preserves_all_examples_no_overlap():
    examples = (
        [{"id": f"tc-{i}", "category": "tool_call"} for i in range(40)]
        + [{"id": f"pc-{i}", "category": "plain_chat"} for i in range(20)]
    )

    train, val, ev = stratified_split(examples, train_ratio=0.8, val_ratio=0.1, eval_ratio=0.1, seed=42)

    assert len(train) + len(val) + len(ev) == len(examples)
    train_ids = {e["id"] for e in train}
    val_ids = {e["id"] for e in val}
    eval_ids = {e["id"] for e in ev}
    assert not (train_ids & val_ids) and not (train_ids & eval_ids) and not (val_ids & eval_ids)


def test_stratified_split_keeps_category_proportions_in_each_split():
    examples = (
        [{"id": f"tc-{i}", "category": "tool_call"} for i in range(100)]
        + [{"id": f"pc-{i}", "category": "plain_chat"} for i in range(100)]
    )

    train, val, ev = stratified_split(examples, train_ratio=0.9, val_ratio=0.05, eval_ratio=0.05, seed=42)

    for split in (train, val, ev):
        tc = sum(1 for e in split if e["category"] == "tool_call")
        pc = sum(1 for e in split if e["category"] == "plain_chat")
        assert abs(tc - pc) <= 1  # balanced input categories -> balanced output


def test_stratified_split_is_deterministic_for_a_fixed_seed():
    examples = [{"id": f"e-{i}", "category": "tool_call"} for i in range(30)]

    run_a = stratified_split(examples, 0.8, 0.1, 0.1, seed=7)
    run_b = stratified_split(examples, 0.8, 0.1, 0.1, seed=7)

    assert [e["id"] for e in run_a[0]] == [e["id"] for e in run_b[0]]


def test_stratified_split_rejects_ratios_that_dont_sum_to_one():
    with pytest.raises(ValueError):
        stratified_split([], train_ratio=0.8, val_ratio=0.1, eval_ratio=0.2, seed=1)


def test_dedupe_examples_drops_duplicate_message_content_keeping_first():
    examples = [
        {"id": "a1", "messages": [{"role": "user", "content": "hi"}], "category": "plain_chat"},
        {"id": "a2", "messages": [{"role": "user", "content": "hi"}], "category": "plain_chat"},
        {"id": "a3", "messages": [{"role": "user", "content": "bye"}], "category": "plain_chat"},
    ]

    result = dedupe_examples(examples)

    assert [e["id"] for e in result] == ["a1", "a3"]


def test_dedupe_examples_keeps_examples_with_same_id_but_different_content():
    examples = [
        {"id": "x", "messages": [{"role": "user", "content": "one"}], "category": "plain_chat"},
        {"id": "x", "messages": [{"role": "user", "content": "two"}], "category": "plain_chat"},
    ]

    result = dedupe_examples(examples)

    assert len(result) == 2



def _tool_call_example(content, ex_id="tc"):
    return {
        "id": ex_id,
        "messages": [
            {"role": "user", "content": "?"},
            {"role": "assistant", "content": content},
        ],
        "category": "tool_call",
    }


def test_filter_malformed_tool_calls_drops_invalid_json_target():
    # Real defect found in hermes-function-calling-v1: a literal '\n' text
    # sequence (not an actual newline) immediately after <tool_call>.
    bad = _tool_call_example('<tool_call>\\n{"name": "f", "arguments": {}}\n</tool_call>')

    assert filter_malformed_tool_calls([bad]) == []


def test_filter_malformed_tool_calls_drops_python_repr_style_values():
    bad = _tool_call_example("<tool_call>\n{'name': 'f', 'arguments': {}}\n</tool_call>")

    assert filter_malformed_tool_calls([bad]) == []


def test_filter_malformed_tool_calls_keeps_well_formed_examples():
    good = _tool_call_example('<tool_call>\n{"name": "f", "arguments": {"x": 1}}\n</tool_call>')

    assert filter_malformed_tool_calls([good]) == [good]


def test_filter_malformed_tool_calls_ignores_plain_chat_examples():
    plain = {"id": "pc", "messages": [{"role": "assistant", "content": "hi"}], "category": "plain_chat"}

    assert filter_malformed_tool_calls([plain]) == [plain]


def test_filter_malformed_tool_calls_requires_every_call_in_a_conversation_well_formed():
    example = {
        "id": "multi",
        "messages": [
            {"role": "assistant", "content": '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>'},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "<tool_call>\n{'name': 'b'}\n</tool_call>"},
        ],
        "category": "tool_call",
    }

    assert filter_malformed_tool_calls([example]) == []