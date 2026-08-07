"""Config for the QLoRA training stage."""
from __future__ import annotations

import dataclasses
from pathlib import Path

from llm_internal.config_utils import load_yaml_dataclass


@dataclasses.dataclass
class TrainConfig:
    base_model: str
    data_dir: str
    output_dir: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    learning_rate: float
    epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int
    checkpoint_every_steps: int
    enable_thinking: bool
    seed: int

    def __post_init__(self) -> None:
        if self.enable_thinking is not False:
            raise ValueError(
                "enable_thinking must be false: the training dataset has no "
                "<think> content, so thinking mode must stay disabled to keep "
                "training and inference on-distribution"
            )


def load_train_config(path: str | Path) -> TrainConfig:
    return load_yaml_dataclass(path, TrainConfig)
