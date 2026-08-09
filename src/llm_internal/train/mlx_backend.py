"""MLX (Apple Silicon) training backend. The config-translation and
data-export helpers below are pure and tested locally without the `mlx`
extra installed; `run_mlx_training` (added in Task 4) is Metal-only and
lazy-imports mlx_lm, so importing this module never requires mlx to be
installed.

mlx-lm's own ChatDataset templates training examples through the tokenizer
without a way to pass enable_thinking=False (see
docs/superpowers/specs/2026-08-07-mlx-support-design.md), so training data
is rendered ourselves via train/sft.py's build_training_text and written in
mlx-lm's `text` dataset format instead of letting mlx-lm apply the chat
template.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from llm_internal.train.config import TrainConfig

_ATTN_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}
_MLP_MODULES = {"gate_proj", "up_proj", "down_proj"}

# Mirrors mlx_lm.lora.CONFIG_DEFAULTS (mlx-lm 0.31.3). Kept as a local
# constant rather than imported from mlx_lm so this module -- and its
# tests -- never require the `mlx` extra to be installed.
_MLX_LORA_CONFIG_DEFAULTS = {
    "model": "Qwen/Qwen3-0.6b",
    "train": False,
    "fine_tune_type": "lora",
    "optimizer": "adam",
    "optimizer_config": {"adam": {}, "adamw": {}, "muon": {}, "sgd": {}, "adafactor": {}},
    "data": "data",
    "seed": 0,
    "num_layers": 16,
    "batch_size": 4,
    "iters": 1000,
    "val_batches": 25,
    "learning_rate": 1e-5,
    "steps_per_report": 10,
    "steps_per_eval": 200,
    "resume_adapter_file": None,
    "adapter_path": "adapters",
    "save_every": 100,
    "test": False,
    "test_batches": 500,
    "max_seq_length": 2048,
    "config": None,
    "grad_checkpoint": False,
    "grad_accumulation_steps": 1,
    "clear_cache_threshold": None,
    "lr_schedule": None,
    "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 20.0},
    "mask_prompt": False,
    "report_to": None,
    "project_name": None,
    "trust_remote_code": False,
}


def lora_scale(lora_alpha: int, lora_r: int) -> float:
    """mlx-lm has no `alpha` concept -- it applies `scale` directly, where
    PEFT/Unsloth compute scale = alpha / r internally. This makes that
    translation explicit."""
    return lora_alpha / lora_r


def target_modules_to_mlx_keys(target_modules: list[str]) -> list[str]:
    """Maps HF/PEFT bare module names (as used in configs/train.yaml's
    target_modules) to mlx-lm's block-relative module paths."""
    keys = []
    for name in target_modules:
        if name in _ATTN_MODULES:
            keys.append(f"self_attn.{name}")
        elif name in _MLP_MODULES:
            keys.append(f"mlp.{name}")
        else:
            raise ValueError(
                f"unrecognized target module {name!r}; expected one of "
                f"{sorted(_ATTN_MODULES | _MLP_MODULES)}"
            )
    return keys


def build_mlx_lora_config(cfg: TrainConfig) -> dict:
    """mlx-lm's `lora_parameters` shape: rank/scale/dropout plus the
    per-module `keys` target list, derived from the same lora_r/lora_alpha/
    lora_dropout/target_modules fields the CUDA backend uses -- no
    duplicated hyperparameter surface."""
    return {
        "rank": cfg.lora_r,
        "scale": lora_scale(cfg.lora_alpha, cfg.lora_r),
        "dropout": cfg.lora_dropout,
        "keys": target_modules_to_mlx_keys(cfg.target_modules),
    }


def compute_mlx_iters(num_examples: int, batch_size: int, epochs: int) -> int:
    """mlx-lm trains for a step count (`iters`), not epochs; this converts
    cfg.epochs at a given batch size into the equivalent step count."""
    if num_examples <= 0 or batch_size <= 0 or epochs <= 0:
        raise ValueError("num_examples, batch_size, and epochs must all be positive")
    steps_per_epoch = math.ceil(num_examples / batch_size)
    return steps_per_epoch * epochs


def build_mlx_training_args(
    cfg: TrainConfig,
    base_dir: str | Path,
    data_dir: str | Path,
    num_train_examples: int,
) -> dict:
    """Overlays TrainConfig onto mlx_lm.lora's CONFIG_DEFAULTS shape. The
    result is passed to mlx_lm.lora.run via types.SimpleNamespace(**args)
    by run_mlx_training (Task 4)."""
    args = dict(_MLX_LORA_CONFIG_DEFAULTS)
    args.update(
        {
            "model": str(base_dir),
            "train": True,
            "data": str(data_dir),
            "adapter_path": str(cfg.output_dir),
            "seed": cfg.seed,
            "num_layers": cfg.mlx_num_layers,
            "batch_size": cfg.per_device_train_batch_size,
            "iters": compute_mlx_iters(num_train_examples, cfg.per_device_train_batch_size, cfg.epochs),
            "learning_rate": cfg.learning_rate,
            "steps_per_eval": cfg.checkpoint_every_steps,
            "save_every": cfg.checkpoint_every_steps,
            "max_seq_length": cfg.max_seq_length,
            "lora_parameters": build_mlx_lora_config(cfg),
        }
    )
    return args


def export_data_for_mlx(data_dir: str | Path, tokenizer, out_dir: str | Path) -> None:
    """Reads train.jsonl/val.jsonl (data/prepare.py output) and writes
    mlx-lm's expected train.jsonl/valid.jsonl `text`-format files into
    out_dir, rendering each conversation via train/sft.py's
    build_training_text so mlx-lm never applies its own chat template
    (see module docstring)."""
    from llm_internal.train.sft import build_training_text, load_split

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in (("train.jsonl", "train.jsonl"), ("val.jsonl", "valid.jsonl")):
        examples = load_split(Path(data_dir) / src_name)
        with open(out_dir / dst_name, "w", encoding="utf-8") as f:
            for ex in examples:
                text = build_training_text(ex["messages"], tokenizer)
                f.write(json.dumps({"text": text}) + "\n")



def run_mlx_training(cfg: TrainConfig) -> None:
    """Metal-only: quantizes the base model to a local 4-bit MLX copy under
    cfg.output_dir/mlx_base (this quantized-base-plus-adapter combination is
    mlx-lm's QLoRA -- there is no separate QLoRA flag), renders the
    train/val splits to mlx-lm's text format under cfg.output_dir/mlx_data
    via export_data_for_mlx, and runs mlx_lm.lora's training loop. Adapters
    are written to cfg.output_dir (adapters.safetensors + adapter_config.json)
    by mlx_lm.lora itself."""
    import types

    from mlx_lm import convert
    from mlx_lm import load as mlx_load
    from mlx_lm.lora import run as mlx_lora_run

    from llm_internal.train.sft import load_split

    output_dir = Path(cfg.output_dir)
    base_dir = output_dir / "mlx_base"
    data_dir = output_dir / "mlx_data"

    if not base_dir.exists():
        convert(cfg.base_model, mlx_path=str(base_dir), quantize=True, q_bits=4, revision=cfg.base_model_revision)

    _, tokenizer = mlx_load(str(base_dir))
    export_data_for_mlx(cfg.data_dir, tokenizer, data_dir)

    train_examples = load_split(Path(cfg.data_dir) / "train.jsonl")
    args_dict = build_mlx_training_args(cfg, base_dir, data_dir, len(train_examples))
    mlx_lora_run(types.SimpleNamespace(**args_dict))