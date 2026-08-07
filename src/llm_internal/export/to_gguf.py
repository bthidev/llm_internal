"""Merge the trained LoRA adapter into the base model, quantize to GGUF, and
write an Ollama Modelfile. The merge/quantize step is GPU-only (Unsloth);
Modelfile rendering is pure and tested locally."""
from __future__ import annotations

from pathlib import Path

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. Call a tool only "
    "when it is necessary to answer the user's request."
)


def render_modelfile(gguf_filename: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return (
        f"FROM ./{gguf_filename}\n"
        f'SYSTEM """{system_prompt}"""\n'
        "PARAMETER temperature 0.7\n"
        "PARAMETER top_p 0.8\n"
        "PARAMETER top_k 20\n"
    )


def write_modelfile(path: str | Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def merge_and_quantize(model_dir: str, output_dir: str, quant: str = "q4_k_m") -> Path:
    """GPU-only: loads the LoRA-adapted model from `model_dir`, merges it into
    the base model, and writes a quantized GGUF into `output_dir` via
    Unsloth's save_pretrained_gguf."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(model_name=model_dir)
    model.save_pretrained_gguf(output_dir, tokenizer, quantization_method=quant)
    matches = sorted(Path(output_dir).glob(f"*{quant}*.gguf"))
    if not matches:
        raise FileNotFoundError(f"no {quant} gguf produced in {output_dir}")
    return matches[0]


def main() -> None:
    from llm_internal.train.config import load_train_config

    train_cfg = load_train_config("configs/train.yaml")
    export_dir = "export"
    gguf_path = merge_and_quantize(train_cfg.output_dir, export_dir)
    write_modelfile(Path(export_dir) / "Modelfile", render_modelfile(gguf_path.name))
    print(f"exported {gguf_path} and Modelfile to {export_dir}")


if __name__ == "__main__":
    main()
