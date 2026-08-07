# tests/test_config_utils.py
import dataclasses
from pathlib import Path

import pytest

from llm_internal.config_utils import load_yaml_dataclass


@dataclasses.dataclass
class _Sample:
    name: str
    count: int


def test_load_yaml_dataclass_constructs_from_yaml(tmp_path: Path):
    path = tmp_path / "sample.yaml"
    path.write_text("name: widget\ncount: 3\n")

    result = load_yaml_dataclass(path, _Sample)

    assert result == _Sample(name="widget", count=3)


def test_load_yaml_dataclass_rejects_unknown_keys(tmp_path: Path):
    path = tmp_path / "sample.yaml"
    path.write_text("name: widget\ncount: 3\nextra: nope\n")

    with pytest.raises(ValueError, match="extra"):
        load_yaml_dataclass(path, _Sample)


def test_load_yaml_dataclass_raises_on_missing_required_field(tmp_path: Path):
    path = tmp_path / "sample.yaml"
    path.write_text("name: widget\n")

    with pytest.raises(TypeError):
        load_yaml_dataclass(path, _Sample)
