"""Config for the dataset preparation stage."""
from __future__ import annotations

import dataclasses
from pathlib import Path

from llm_internal.config_utils import load_yaml_dataclass


@dataclasses.dataclass
class DataConfig:
    dataset_repo: str
    dataset_revision: str
    dataset_files: list[str]
    output_dir: str
    train_ratio: float
    val_ratio: float
    eval_ratio: float
    seed: int


def load_data_config(path: str | Path) -> DataConfig:
    return load_yaml_dataclass(path, DataConfig)
