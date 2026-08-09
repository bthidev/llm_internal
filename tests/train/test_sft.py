# tests/train/test_sft.py
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from llm_internal.data.prepare import write_jsonl
from llm_internal.train.config import TrainConfig
from llm_internal.train.sft import build_hf_dataset, build_training_text, load_split, run_training


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


def _full_train_config(**overrides) -> TrainConfig:
    values = dict(
        base_model="Qwen/Qwen3-1.7B",
        base_model_revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        data_dir="data/processed",
        output_dir="checkpoints",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        learning_rate=0.0002,
        epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_seq_length=2048,
        checkpoint_every_steps=50,
        enable_thinking=False,
        backend="cuda",
        mlx_num_layers=-1,
        seed=42,
    )
    values.update(overrides)
    return TrainConfig(**values)


def test_run_training_dispatches_to_mlx_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "llm_internal.train.mlx_backend.run_mlx_training",
        lambda cfg: calls.append(cfg),
    )
    cfg = _full_train_config(backend="mlx")

    run_training(cfg)

    assert calls == [cfg]


def test_run_training_dispatches_to_unsloth_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "llm_internal.train.sft._run_training_unsloth",
        lambda cfg: calls.append(cfg),
    )
    cfg = _full_train_config(backend="cuda")

    run_training(cfg)

    assert calls == [cfg]
