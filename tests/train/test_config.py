# tests/train/test_config.py
from pathlib import Path

import pytest

from llm_internal.train.config import TrainConfig, load_train_config


def test_load_train_config_reads_real_config_file():
    cfg = load_train_config("configs/train.yaml")

    assert isinstance(cfg, TrainConfig)
    assert cfg.base_model == "Qwen/Qwen3-1.7B"
    assert cfg.enable_thinking is False
    assert cfg.lora_r > 0
    assert isinstance(cfg.target_modules, list) and cfg.target_modules
    assert cfg.backend == "cuda"
    assert cfg.mlx_num_layers == -1


def test_load_train_config_rejects_enable_thinking_true(tmp_path: Path):
    bad = tmp_path / "train.yaml"
    bad.write_text(
        "base_model: Qwen/Qwen3-1.7B\n"
        "base_model_revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e\n"
        "data_dir: data/processed\n"
        "output_dir: checkpoints\n"
        "lora_r: 16\n"
        "lora_alpha: 32\n"
        "lora_dropout: 0.05\n"
        "target_modules: [q_proj, v_proj]\n"
        "learning_rate: 0.0002\n"
        "epochs: 1\n"
        "per_device_train_batch_size: 2\n"
        "gradient_accumulation_steps: 4\n"
        "max_seq_length: 2048\n"
        "checkpoint_every_steps: 50\n"
        "enable_thinking: true\n"
        "backend: cuda\n"
        "mlx_num_layers: -1\n"
        "seed: 42\n"
    )

    with pytest.raises(ValueError, match="enable_thinking"):
        load_train_config(bad)


def test_load_train_config_rejects_invalid_backend(tmp_path: Path):
    bad = tmp_path / "train.yaml"
    bad.write_text(
        "base_model: Qwen/Qwen3-1.7B\n"
        "base_model_revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e\n"
        "data_dir: data/processed\n"
        "output_dir: checkpoints\n"
        "lora_r: 16\n"
        "lora_alpha: 32\n"
        "lora_dropout: 0.05\n"
        "target_modules: [q_proj, v_proj]\n"
        "learning_rate: 0.0002\n"
        "epochs: 1\n"
        "per_device_train_batch_size: 2\n"
        "gradient_accumulation_steps: 4\n"
        "max_seq_length: 2048\n"
        "checkpoint_every_steps: 50\n"
        "enable_thinking: false\n"
        "backend: tpu\n"
        "mlx_num_layers: -1\n"
        "seed: 42\n"
    )

    with pytest.raises(ValueError, match="backend"):
        load_train_config(bad)
