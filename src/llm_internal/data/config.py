"""Config for the dataset preparation stage."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml


@dataclasses.dataclass
class SourceConfig:
    """One raw dataset to download and format. `format` selects the transform
    in `llm_internal.data.transform`/`prepare._format_raw_example`:
    `"hermes"` (ShareGPT-style `conversations`, already tool-call-tagged),
    `"alpaca_code"` (flat `instruction`/`input`/`output`, no tools), or
    `"glaive"` (`system`/`chat` text blobs with inline function defs and
    `<functioncall>`/`FUNCTION RESPONSE` markers).
    """

    name: str
    format: str
    dataset_repo: str
    dataset_revision: str
    dataset_files: list[str]


@dataclasses.dataclass
class DataConfig:
    sources: list[SourceConfig]
    output_dir: str
    train_ratio: float
    val_ratio: float
    eval_ratio: float
    seed: int


def load_data_config(path: str | Path) -> DataConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    sources_raw = raw.pop("sources", [])
    sources = [SourceConfig(**s) for s in sources_raw]

    field_names = {field.name for field in dataclasses.fields(DataConfig)} - {"sources"}
    unknown = set(raw) - field_names
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")

    return DataConfig(sources=sources, **raw)
