# tests/export/test_run_export.py
from pathlib import Path

from llm_internal.export.config import ExportConfig
from llm_internal.export.run_export import run_export


def test_run_export_dispatches_to_mlx(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "llm_internal.export.to_mlx.fuse_and_quantize_mlx",
        lambda model_dir, output_dir, q_bits=4: calls.append(("fuse", model_dir, output_dir)) or Path(output_dir),
    )
    monkeypatch.setattr(
        "llm_internal.export.to_mlx.write_readme",
        lambda path, content: calls.append(("readme", str(path))),
    )
    cfg = ExportConfig(backend="mlx", model_dir=str(tmp_path / "m"), output_dir=str(tmp_path / "o"), quant="4bit")

    result = run_export(cfg)

    assert result == Path(str(tmp_path / "o"))
    assert calls[0][0] == "fuse"
    assert calls[1][0] == "readme"


def test_run_export_dispatches_to_cuda(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "llm_internal.export.to_gguf.merge_and_quantize",
        lambda model_dir, output_dir, quant: calls.append(("merge", model_dir, output_dir, quant)) or Path(output_dir) / "model-q4_k_m.gguf",
    )
    monkeypatch.setattr(
        "llm_internal.export.to_gguf.write_modelfile",
        lambda path, content: calls.append(("modelfile", str(path))),
    )
    cfg = ExportConfig(backend="cuda", model_dir=str(tmp_path / "m"), output_dir=str(tmp_path / "o"), quant="q4_k_m")

    result = run_export(cfg)

    assert result == Path(str(tmp_path / "o")) / "model-q4_k_m.gguf"
    assert calls[0][0] == "merge"
    assert calls[1][0] == "modelfile"
