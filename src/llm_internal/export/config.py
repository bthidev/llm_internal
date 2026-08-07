"""Config for the export stage: GGUF/Ollama for the cuda backend, a fused
+ quantized MLX weights directory for the mlx backend."""
from __future__ import annotations

import dataclasses
from pathlib import Path

from llm_internal.config_utils import load_yaml_dataclass


@dataclasses.dataclass
class ExportConfig:
    backend: str
    model_dir: str
    output_dir: str
    quant: str

    def __post_init__(self) -> None:
        if self.backend not in ("cuda", "mlx"):
            raise ValueError(f"backend must be 'cuda' or 'mlx', got {self.backend!r}")


def load_export_config(path: str | Path) -> ExportConfig:
    return load_yaml_dataclass(path, ExportConfig)
