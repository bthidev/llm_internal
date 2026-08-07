import pytest

from llm_internal.export.config import ExportConfig, load_export_config


def test_load_export_config_reads_real_config_file():
    cfg = load_export_config("configs/export.yaml")

    assert isinstance(cfg, ExportConfig)
    assert cfg.backend == "cuda"
    assert cfg.model_dir == "checkpoints"
    assert cfg.output_dir == "export"
    assert cfg.quant == "q4_k_m"


def test_export_config_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        ExportConfig(backend="tpu", model_dir="m", output_dir="o", quant="q")
