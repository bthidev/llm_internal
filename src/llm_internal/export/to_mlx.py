"""MLX export: fuse the LoRA adapter into its quantized base and re-quantize
via mlx-lm (Metal-only), and render a README documenting how to run the
result -- there is no MLX equivalent of a GGUF Modelfile/Ollama."""
from __future__ import annotations

from pathlib import Path

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. Call a tool only "
    "when it is necessary to answer the user's request."
)


def render_mlx_readme(output_dir: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return (
        "# homemade-llm (MLX)\n\n"
        f"System prompt used during fine-tuning: {system_prompt}\n\n"
        "Run with mlx-lm (`uv sync --extra mlx` on Apple Silicon):\n\n"
        "```bash\n"
        f"mlx_lm.generate --model {output_dir} "
        '--chat-template-args \'{"enable_thinking": false}\' '
        '--prompt "your prompt here"\n'
        f"mlx_lm.server --model {output_dir}\n"
        "```\n"
    )


def write_readme(path: str | Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def fuse_and_quantize_mlx(model_dir: str, output_dir: str, q_bits: int = 4) -> Path:
    """Metal-only: model_dir is a run_mlx_training output directory
    (adapters.safetensors + adapter_config.json + a mlx_base/ subdirectory
    holding the quantized base -- see train/mlx_backend.py run_mlx_training).
    Fuses the adapter into mlx_base via the mlx_lm.fuse console script, then
    re-quantizes the fused model to q_bits via mlx_lm.convert.convert.
    Returns output_dir.
    """
    import subprocess

    from mlx_lm import convert

    base_dir = Path(model_dir) / "mlx_base"
    fused_dir = Path(output_dir) / "_fused"
    subprocess.run(
        [
            "mlx_lm.fuse",
            "--model", str(base_dir),
            "--adapter-path", str(model_dir),
            "--save-path", str(fused_dir),
        ],
        check=True,
    )
    convert(str(fused_dir), mlx_path=output_dir, quantize=True, q_bits=q_bits)
    return Path(output_dir)
