# tests/train/test_sft.py
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from llm_internal.train.sft import build_hf_dataset, build_training_text, load_split
from llm_internal.data.prepare import write_jsonl


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")


def test_build_training_text_renders_tool_call_content_verbatim(tokenizer):
    messages = [
        {"role": "system", "content": "sys<tools>[]</tools>"},
        {"role": "user", "content": "what's the weather?"},
        {"role": "assistant", "content": '<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>'},
    ]

    text = build_training_text(messages, tokenizer)

    assert "<tool_call>" in text
    assert '"name": "f"' in text
    assert text.count("<|im_start|>") == 3


def test_build_training_text_forces_non_thinking_final_turn(tokenizer):
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    text = build_training_text(messages, tokenizer)

    # Qwen3's template auto-inserts an empty think block on the final
    # assistant turn of a full conversation; no reasoning content leaks in.
    assert "<think>\n\n</think>" in text


def test_load_split_reads_jsonl(tmp_path: Path):
    examples = [{"id": "a", "messages": [], "category": "plain_chat"}]
    path = tmp_path / "train.jsonl"
    write_jsonl(examples, path)

    result = load_split(path)

    assert result == examples


def test_build_hf_dataset_adds_text_column(tokenizer):
    examples = [
        {"id": "a", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}], "category": "plain_chat"},
    ]

    ds = build_hf_dataset(examples, tokenizer)

    assert len(ds) == 1
    assert "text" in ds.column_names
    assert "<|im_start|>" in ds[0]["text"]
