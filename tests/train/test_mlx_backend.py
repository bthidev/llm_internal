import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from llm_internal.data.prepare import write_jsonl
from llm_internal.train.config import TrainConfig
from llm_internal.train.mlx_backend import (
    build_mlx_lora_config,
    build_mlx_training_args,
    compute_mlx_iters,
    export_data_for_mlx,
    lora_scale,
    target_modules_to_mlx_keys,
)


def _cfg(**overrides) -> TrainConfig:
    values = dict(
        base_model="Qwen/Qwen3-1.7B",
        data_dir="data/processed",
        output_dir="checkpoints",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        learning_rate=0.0002,
        epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        max_seq_length=4096,
        checkpoint_every_steps=50,
        enable_thinking=False,
        backend="mlx",
        mlx_num_layers=-1,
        seed=42,
    )
    values.update(overrides)
    return TrainConfig(**values)


def test_lora_scale_divides_alpha_by_rank():
    assert lora_scale(lora_alpha=32, lora_r=16) == 2.0


def test_target_modules_to_mlx_keys_maps_attention_and_mlp_names():
    keys = target_modules_to_mlx_keys(["q_proj", "gate_proj"])

    assert keys == ["self_attn.q_proj", "mlp.gate_proj"]


def test_target_modules_to_mlx_keys_rejects_unknown_module():
    with pytest.raises(ValueError, match="unrecognized"):
        target_modules_to_mlx_keys(["not_a_real_module"])


def test_build_mlx_lora_config_derives_scale_and_keys():
    cfg = _cfg(lora_r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"])

    result = build_mlx_lora_config(cfg)

    assert result == {"rank": 16, "scale": 2.0, "dropout": 0.1, "keys": ["self_attn.q_proj", "self_attn.v_proj"]}


def test_compute_mlx_iters_multiplies_steps_per_epoch_by_epochs():
    assert compute_mlx_iters(num_examples=100, batch_size=8, epochs=3) == 39  # ceil(100/8)=13, *3


def test_compute_mlx_iters_rejects_non_positive_inputs():
    with pytest.raises(ValueError):
        compute_mlx_iters(num_examples=0, batch_size=8, epochs=3)


def test_build_mlx_training_args_overlays_cfg_onto_defaults():
    cfg = _cfg()

    args = build_mlx_training_args(cfg, base_dir="checkpoints/mlx_base", data_dir="checkpoints/mlx_data", num_train_examples=100)

    assert args["model"] == "checkpoints/mlx_base"
    assert args["train"] is True
    assert args["data"] == "checkpoints/mlx_data"
    assert args["adapter_path"] == "checkpoints"
    assert args["num_layers"] == -1
    assert args["batch_size"] == 2
    assert args["iters"] == compute_mlx_iters(100, 2, 3)
    assert args["lora_parameters"] == build_mlx_lora_config(cfg)
    # untouched defaults survive the overlay
    assert args["optimizer"] == "adam"
    assert args["fine_tune_type"] == "lora"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")


def test_export_data_for_mlx_writes_text_format_train_and_valid_files(tmp_path: Path, tokenizer):
    data_dir = tmp_path / "processed"
    write_jsonl(
        [{"id": "a", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}], "category": "plain_chat"}],
        data_dir / "train.jsonl",
    )
    write_jsonl(
        [{"id": "b", "messages": [{"role": "user", "content": "yo"}, {"role": "assistant", "content": "sup"}], "category": "plain_chat"}],
        data_dir / "val.jsonl",
    )
    out_dir = tmp_path / "mlx_data"

    export_data_for_mlx(data_dir, tokenizer, out_dir)

    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "valid.jsonl").exists()
    train_line = json.loads((out_dir / "train.jsonl").read_text().strip().splitlines()[0])
    assert "text" in train_line
    assert "<|im_start|>" in train_line["text"]
