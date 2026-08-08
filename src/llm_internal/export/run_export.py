"""Config-driven export dispatcher: routes to the GGUF/Ollama (cuda) or
MLX (mlx) export path based on ExportConfig.backend. export/to_gguf.py's
own main() (CUDA-only, reads configs/train.yaml directly) is left
unchanged for direct standalone invocation; this module is the new
config-driven entrypoint both backends go through."""
from __future__ import annotations

from pathlib import Path

from llm_internal.export.config import ExportConfig, load_export_config


def run_export(cfg: ExportConfig) -> Path:
    if cfg.backend == "mlx":
        from llm_internal.export.to_mlx import fuse_and_quantize_mlx, render_mlx_readme, write_readme

        out = fuse_and_quantize_mlx(cfg.model_dir, cfg.output_dir)
        write_readme(Path(cfg.output_dir) / "README.md", render_mlx_readme(cfg.output_dir))
        return out

    from llm_internal.export.to_gguf import merge_and_quantize, render_modelfile, write_modelfile

    gguf_path = merge_and_quantize(cfg.model_dir, cfg.output_dir, cfg.quant)
    write_modelfile(Path(cfg.output_dir) / "Modelfile", render_modelfile(gguf_path.name))
    return gguf_path


def main() -> None:
    cfg = load_export_config("configs/export.yaml")
    result = run_export(cfg)
    print(f"exported ({cfg.backend}) to {result}")


if __name__ == "__main__":
    main()
