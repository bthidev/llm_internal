"""Shared YAML -> dataclass config loading, used by data/train/eval configs."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Type, TypeVar

import yaml

T = TypeVar("T")


def load_yaml_dataclass(path: str | Path, cls: Type[T]) -> T:
    """Load a YAML file's top-level mapping and construct `cls` (a dataclass)
    from it. Raises ValueError on unknown keys, TypeError (via the dataclass
    constructor) on missing required fields.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    field_names = {field.name for field in dataclasses.fields(cls)}
    unknown = set(raw) - field_names
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")

    return cls(**raw)
