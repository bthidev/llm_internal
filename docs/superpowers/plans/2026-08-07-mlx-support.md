# MLX Backend Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Apple Silicon (MLX) as a second, config-selected backend for training, eval generation, and export, alongside the existing CUDA/Unsloth backend, so the pipeline can run end-to-end on a Mac.

**Architecture:** `TrainConfig`/`EvalConfig`/`ExportConfig` each gain a `backend: "cuda"|"mlx"` field. Each stage's entrypoint (`train.sft.run_training`, `eval.run_eval._load_and_generate`, `export.run_export.run_export`) becomes a thin dispatcher to either the existing CUDA implementation (unchanged) or a new MLX implementation (`train/mlx_backend.py`, `eval/run_eval.generate_predictions_mlx`, `export/to_mlx.py`). Config translation, LoRA hyperparameter mapping, and training-data rendering for MLX are pure/tested-locally; the actual `mlx_lm` calls are Metal-only and lazy-imported, exercised on real Apple Silicon hardware via `scripts/run_on_mac_mlx.sh`, mirroring how the CUDA path is untestable on this workstation and exercised via `scripts/run_on_runpod.sh`.

**Tech Stack:** Python 3.10+, `mlx-lm>=0.31.3` (new optional dependency group `mlx`), same stack as the base plan otherwise.

## Global Constraints

- Base model, dataset, and split constraints from `docs/superpowers/plans/2026-08-07-homemade-llm.md` are unchanged and apply identically to both backends.
- `enable_thinking` must stay `False` end to end. mlx-lm's `mlx_lm.lora` training-data templater cannot be told `enable_thinking=False` (confirmed from source: `ChatDataset.process` never passes it, so the wrapper always injects `enable_thinking=True` for Qwen3). Training data for MLX is therefore rendered ourselves via the existing `build_training_text` (never let mlx-lm apply its own chat template for training). For MLX eval/inference, `enable_thinking=False` is passed explicitly — that override is supported there.
- mlx-lm cannot produce GGUF for Qwen3 (`mlx_lm.fuse --export-gguf` is F16-only and restricted to `llama`/`mixtral`/`mistral` model types). The MLX export path never attempts GGUF; it produces a fused+quantized MLX weights directory instead.
- `mlx-lm` (and the `mlx` package it depends on) only builds on Apple Silicon macOS. It must be an optional dependency (`[project.optional-dependencies] mlx`), never in the default/dev install, and every module that touches `mlx_lm` must lazy-import it inside function bodies so importing the module never requires `mlx` to be installed.
- No duplicated hyperparameter surface: MLX's `scale` and `keys` are derived from the existing `lora_alpha`/`lora_r`/`target_modules` fields, not new independent config values.
- Backend fields (`backend` on `TrainConfig`/`EvalConfig`/`ExportConfig`) reject any value other than `"cuda"`/`"mlx"` at load time via `__post_init__`, matching the existing `enable_thinking` validation style in `TrainConfig`.
- Do not invent project-wide lint/format commands — run only the pytest commands named in each task.

---

### Task 1: `mlx` optional dependency group

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `uv sync --extra dev --extra mlx` installs `mlx-lm`; `uv sync --extra dev` (default) does not.

- [ ] **Step 1: Modify `pyproject.toml`**

```toml
[project]
name = "llm-internal"
version = "0.1.0"
description = "Homemade LLM: QLoRA tool-use fine-tune of Qwen3-1.7B"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.3",
    "transformers>=4.51.0",
    "trl>=0.9.6",
    "peft>=0.11.1",
    "datasets>=2.19",
    "huggingface_hub>=0.23",
    "bitsandbytes>=0.43; sys_platform == 'linux'",
    "unsloth>=2024.8",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]
mlx = ["mlx-lm>=0.31.3"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/llm_internal"]
```

- [ ] **Step 2: Verify the default install is unaffected**

Run: `uv sync --extra dev && uv run python -c "print('ok')"`
Expected: exits 0, prints `ok` (the new `mlx` extra group is not installed by default, no dependency resolution errors)

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add optional mlx dependency group"
```

---

### Task 2: `TrainConfig` backend field

**Files:**
- Modify: `src/llm_internal/train/config.py`
- Modify: `configs/train.yaml`
- Modify: `tests/train/test_config.py`

**Interfaces:**
- Produces: `TrainConfig.backend: str` (validated `"cuda"|"mlx"`), `TrainConfig.mlx_num_layers: int` (`-1` = all layers). Both required fields, no silent defaults, consistent with every other `TrainConfig` field.

- [ ] **Step 1: Write the failing tests**

```python
# tests/train/test_config.py
from pathlib import Path

import pytest

from llm_internal.train.config import TrainConfig, load_train_config


def test_load_train_config_reads_real_config_file():
    cfg = load_train_config("configs/train.yaml")

    assert isinstance(cfg, TrainConfig)
    assert cfg.base_model == "Qwen/Qwen3-1.7B"
    assert cfg.enable_thinking is False
    assert cfg.lora_r > 0
    assert isinstance(cfg.target_modules, list) and cfg.target_modules
    assert cfg.backend == "cuda"
    assert cfg.mlx_num_layers == -1


def test_load_train_config_rejects_enable_thinking_true(tmp_path: Path):
    bad = tmp_path / "train.yaml"
    bad.write_text(
        "base_model: Qwen/Qwen3-1.7B\n"
        "data_dir: data/processed\n"
        "output_dir: checkpoints\n"
        "lora_r: 16\n"
        "lora_alpha: 32\n"
        "lora_dropout: 0.05\n"
        "target_modules: [q_proj, v_proj]\n"
        "learning_rate: 0.0002\n"
        "epochs: 1\n"
        "per_device_train_batch_size: 2\n"
        "gradient_accumulation_steps: 4\n"
        "max_seq_length: 2048\n"
        "checkpoint_every_steps: 50\n"
        "enable_thinking: true\n"
        "backend: cuda\n"
        "mlx_num_layers: -1\n"
        "seed: 42\n"
    )

    with pytest.raises(ValueError, match="enable_thinking"):
        load_train_config(bad)


def test_load_train_config_rejects_invalid_backend(tmp_path: Path):
    bad = tmp_path / "train.yaml"
    bad.write_text(
        "base_model: Qwen/Qwen3-1.7B\n"
        "data_dir: data/processed\n"
        "output_dir: checkpoints\n"
        "lora_r: 16\n"
        "lora_alpha: 32\n"
        "lora_dropout: 0.05\n"
        "target_modules: [q_proj, v_proj]\n"
        "learning_rate: 0.0002\n"
        "epochs: 1\n"
        "per_device_train_batch_size: 2\n"
        "gradient_accumulation_steps: 4\n"
        "max_seq_length: 2048\n"
        "checkpoint_every_steps: 50\n"
        "enable_thinking: false\n"
        "backend: tpu\n"
        "mlx_num_layers: -1\n"
        "seed: 42\n"
    )

    with pytest.raises(ValueError, match="backend"):
        load_train_config(bad)
```

- [ ] **Step 2: Run to verify the new/changed tests fail**

Run: `uv run pytest tests/train/test_config.py -v`
Expected: FAIL — `test_load_train_config_rejects_invalid_backend` fails (no such validation yet); the other two fail with `TypeError`/`AssertionError` (config.py doesn't accept `backend`/`mlx_num_layers` yet)

- [ ] **Step 3: Modify `configs/train.yaml`**

```yaml
base_model: Qwen/Qwen3-1.7B
data_dir: data/processed
output_dir: checkpoints
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
learning_rate: 0.0002
epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
max_seq_length: 4096
checkpoint_every_steps: 50
enable_thinking: false
backend: cuda
mlx_num_layers: -1
seed: 42
```

- [ ] **Step 4: Modify `src/llm_internal/train/config.py`**

```python
"""Config for the QLoRA training stage."""
from __future__ import annotations

import dataclasses
from pathlib import Path

from llm_internal.config_utils import load_yaml_dataclass


@dataclasses.dataclass
class TrainConfig:
    base_model: str
    data_dir: str
    output_dir: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    learning_rate: float
    epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int
    checkpoint_every_steps: int
    enable_thinking: bool
    backend: str
    mlx_num_layers: int
    seed: int

    def __post_init__(self) -> None:
        if self.enable_thinking is not False:
            raise ValueError(
                "enable_thinking must be false: the training dataset has no "
                "<think> content, so thinking mode must stay disabled to keep "
                "training and inference on-distribution"
            )
        if self.backend not in ("cuda", "mlx"):
            raise ValueError(f"backend must be 'cuda' or 'mlx', got {self.backend!r}")


def load_train_config(path: str | Path) -> TrainConfig:
    return load_yaml_dataclass(path, TrainConfig)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/train/test_config.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add src/llm_internal/train/config.py configs/train.yaml tests/train/test_config.py
git commit -m "feat: add backend field to TrainConfig"
```

---

### Task 3: MLX config/data helpers (pure)

**Files:**
- Create: `src/llm_internal/train/mlx_backend.py`
- Test: `tests/train/test_mlx_backend.py`

**Interfaces:**
- Consumes: `TrainConfig` (Task 2), `build_training_text`/`load_split` (`train/sft.py`, existing), `write_jsonl` (`data/prepare.py`, existing).
- Produces: `lora_scale(lora_alpha, lora_r) -> float`, `target_modules_to_mlx_keys(target_modules: list[str]) -> list[str]`, `build_mlx_lora_config(cfg: TrainConfig) -> dict`, `compute_mlx_iters(num_examples, batch_size, epochs) -> int`, `build_mlx_training_args(cfg, base_dir, data_dir, num_train_examples) -> dict`, `export_data_for_mlx(data_dir, tokenizer, out_dir) -> None`. All pure/testable without `mlx` installed — this module never imports `mlx_lm` at module scope.

- [ ] **Step 1: Write the failing tests**

```python
# tests/train/test_mlx_backend.py
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from llm_internal.data.prepare import write_jsonl
from llm_internal.train.config import TrainConfig
from llm_internal.train.mlx_backend import (
    build_mlx_lora_config,
    build_mlx_training_args,
    compute_mlx_iters,
    export_data_for_mlx,
    lora_scale,
    target_modules_to_mlx_keys,
)


def _cfg(**overrides) -> TrainConfig:
    values = dict(
        base_model="Qwen/Qwen3-1.7B",
        data_dir="data/processed",
        output_dir="checkpoints",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        learning_rate=0.0002,
        epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        max_seq_length=4096,
        checkpoint_every_steps=50,
        enable_thinking=False,
        backend="mlx",
        mlx_num_layers=-1,
        seed=42,
    )
    values.update(overrides)
    return TrainConfig(**values)


def test_lora_scale_divides_alpha_by_rank():
    assert lora_scale(lora_alpha=32, lora_r=16) == 2.0


def test_target_modules_to_mlx_keys_maps_attention_and_mlp_names():
    keys = target_modules_to_mlx_keys(["q_proj", "gate_proj"])

    assert keys == ["self_attn.q_proj", "mlp.gate_proj"]


def test_target_modules_to_mlx_keys_rejects_unknown_module():
    with pytest.raises(ValueError, match="unrecognized"):
        target_modules_to_mlx_keys(["not_a_real_module"])


def test_build_mlx_lora_config_derives_scale_and_keys():
    cfg = _cfg(lora_r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"])

    result = build_mlx_lora_config(cfg)

    assert result == {"rank": 16, "scale": 2.0, "dropout": 0.1, "keys": ["self_attn.q_proj", "self_attn.v_proj"]}


def test_compute_mlx_iters_multiplies_steps_per_epoch_by_epochs():
    assert compute_mlx_iters(num_examples=100, batch_size=8, epochs=3) == 39  # ceil(100/8)=13, *3


def test_compute_mlx_iters_rejects_non_positive_inputs():
    with pytest.raises(ValueError):
        compute_mlx_iters(num_examples=0, batch_size=8, epochs=3)


def test_build_mlx_training_args_overlays_cfg_onto_defaults():
    cfg = _cfg()

    args = build_mlx_training_args(cfg, base_dir="checkpoints/mlx_base", data_dir="checkpoints/mlx_data", num_train_examples=100)

    assert args["model"] == "checkpoints/mlx_base"
    assert args["train"] is True
    assert args["data"] == "checkpoints/mlx_data"
    assert args["adapter_path"] == "checkpoints"
    assert args["num_layers"] == -1
    assert args["batch_size"] == 2
    assert args["iters"] == compute_mlx_iters(100, 2, 3)
    assert args["lora_parameters"] == build_mlx_lora_config(cfg)
    # untouched defaults survive the overlay
    assert args["optimizer"] == "adam"
    assert args["fine_tune_type"] == "lora"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")


def test_export_data_for_mlx_writes_text_format_train_and_valid_files(tmp_path: Path, tokenizer):
    data_dir = tmp_path / "processed"
    write_jsonl(
        [{"id": "a", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}], "category": "plain_chat"}],
        data_dir / "train.jsonl",
    )
    write_jsonl(
        [{"id": "b", "messages": [{"role": "user", "content": "yo"}, {"role": "assistant", "content": "sup"}], "category": "plain_chat"}],
        data_dir / "val.jsonl",
    )
    out_dir = tmp_path / "mlx_data"

    export_data_for_mlx(data_dir, tokenizer, out_dir)

    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "valid.jsonl").exists()
    train_line = json.loads((out_dir / "train.jsonl").read_text().strip().splitlines()[0])
    assert "text" in train_line
    assert "<|im_start|>" in train_line["text"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/train/test_mlx_backend.py -v`
Expected: FAIL / ImportError — `llm_internal.train.mlx_backend` does not exist yet

- [ ] **Step 3: Write `src/llm_internal/train/mlx_backend.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/train/test_mlx_backend.py -v`
Expected: PASS (8 tests) — first run downloads the Qwen3-1.7B tokenizer if not already cached from Task 5 of the base plan

- [ ] **Step 5: Commit**

```bash
git add src/llm_internal/train/mlx_backend.py tests/train/test_mlx_backend.py
git commit -m "feat: add MLX LoRA config and training-data export helpers"
```

---

### Task 4: MLX training entrypoint + `run_training` dispatcher

**Files:**
- Modify: `src/llm_internal/train/mlx_backend.py`
- Modify: `src/llm_internal/train/sft.py`
- Modify: `tests/train/test_sft.py`

**Interfaces:**
- Consumes: `build_mlx_training_args`, `export_data_for_mlx` (Task 3); `TrainConfig` (Task 2).
- Produces: `run_mlx_training(cfg: TrainConfig) -> None` (Metal-only, in `mlx_backend.py`); `train/sft.py`'s `run_training(cfg)` becomes a dispatcher, existing Unsloth body moves to `_run_training_unsloth(cfg)` unchanged.

- [ ] **Step 1: Write the failing dispatcher tests**

```python
# tests/train/test_sft.py
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from llm_internal.data.prepare import write_jsonl
from llm_internal.train.config import TrainConfig
from llm_internal.train.sft import build_hf_dataset, build_training_text, load_split, run_training


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")


def test_build_training_text_renders_tool_call_content_verbatim(tokenizer):
    messages = [
        {"role": "system", "content": "sys<tools>[]</tools>"},
        {"role": "user", "content": "what's the weather?"},
        {"role": "assistant", "content": '<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>'},
    ]

    text = build_training_text(messages, tokenizer)

    assert "<tool_call>" in text
    assert '"name": "f"' in text
    assert text.count("<|im_start|>") == 3


def test_build_training_text_forces_non_thinking_final_turn(tokenizer):
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    text = build_training_text(messages, tokenizer)

    assert "<think>\n\n</think>" in text


def test_load_split_reads_jsonl(tmp_path: Path):
    examples = [{"id": "a", "messages": [], "category": "plain_chat"}]
    path = tmp_path / "train.jsonl"
    write_jsonl(examples, path)

    result = load_split(path)

    assert result == examples


def test_build_hf_dataset_adds_text_column(tokenizer):
    examples = [
        {"id": "a", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}], "category": "plain_chat"},
    ]

    ds = build_hf_dataset(examples, tokenizer)

    assert len(ds) == 1
    assert "text" in ds.column_names
    assert "<|im_start|>" in ds[0]["text"]


def _full_train_config(**overrides) -> TrainConfig:
    values = dict(
        base_model="Qwen/Qwen3-1.7B",
        data_dir="data/processed",
        output_dir="checkpoints",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        learning_rate=0.0002,
        epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_seq_length=2048,
        checkpoint_every_steps=50,
        enable_thinking=False,
        backend="cuda",
        mlx_num_layers=-1,
        seed=42,
    )
    values.update(overrides)
    return TrainConfig(**values)


def test_run_training_dispatches_to_mlx_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "llm_internal.train.mlx_backend.run_mlx_training",
        lambda cfg: calls.append(cfg),
    )
    cfg = _full_train_config(backend="mlx")

    run_training(cfg)

    assert calls == [cfg]


def test_run_training_dispatches_to_unsloth_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "llm_internal.train.sft._run_training_unsloth",
        lambda cfg: calls.append(cfg),
    )
    cfg = _full_train_config(backend="cuda")

    run_training(cfg)

    assert calls == [cfg]
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `uv run pytest tests/train/test_sft.py -v`
Expected: FAIL — `run_training` doesn't dispatch yet, `_run_training_unsloth` doesn't exist as a separate name, `mlx_backend.run_mlx_training` doesn't exist

- [ ] **Step 3: Modify `src/llm_internal/train/sft.py`**

```python
"""QLoRA SFT training stage. Chat rendering/dataset assembly are pure and
tested locally; run_training dispatches to the CUDA (Unsloth) or MLX
backend based on TrainConfig.backend. Both backend implementations are
GPU/Metal-only and exercised on real hardware, not in this test suite (see
scripts/run_on_runpod.sh and scripts/run_on_mac_mlx.sh)."""
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from llm_internal.train.config import TrainConfig, load_train_config


def build_training_text(messages: list[dict], tokenizer) -> str:
    """Render a full conversation (list of {"role","content"}) through the
    model's chat template for training. Uses add_generation_prompt=False
    since the whole conversation, including the final assistant reply, is
    present -- no enable_thinking flag is needed here (it only affects the
    generation-prompt-only branch of Qwen3's template).
    """
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def load_split(path: str | Path) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def build_hf_dataset(examples: list[dict], tokenizer) -> Dataset:
    texts = [build_training_text(ex["messages"], tokenizer) for ex in examples]
    return Dataset.from_dict({"text": texts})


def run_training(cfg: TrainConfig) -> None:
    """Dispatches to the CUDA/Unsloth or MLX training backend based on
    cfg.backend."""
    if cfg.backend == "mlx":
        from llm_internal.train.mlx_backend import run_mlx_training

        run_mlx_training(cfg)
    else:
        _run_training_unsloth(cfg)


def _run_training_unsloth(cfg: TrainConfig) -> None:
    """GPU-only: loads Qwen3-1.7B in 4-bit via Unsloth, attaches a LoRA
    adapter, and runs TRL's SFTTrainer. Resumes automatically from the
    latest checkpoint in cfg.output_dir if one exists.
    """
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        random_state=cfg.seed,
    )

    train_examples = load_split(Path(cfg.data_dir) / "train.jsonl")
    val_examples = load_split(Path(cfg.data_dir) / "val.jsonl")
    train_ds = build_hf_dataset(train_examples, tokenizer)
    val_ds = build_hf_dataset(val_examples, tokenizer)

    output_dir = Path(cfg.output_dir)
    resume = output_dir.exists() and any(output_dir.glob("checkpoint-*"))

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        save_steps=cfg.checkpoint_every_steps,
        eval_strategy="steps",
        eval_steps=cfg.checkpoint_every_steps,
        max_seq_length=cfg.max_seq_length,
        dataset_text_field="text",
        seed=cfg.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_dir))


def main() -> None:
    cfg = load_train_config("configs/train.yaml")
    run_training(cfg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Append `run_mlx_training` to `src/llm_internal/train/mlx_backend.py`**

Add this function at the end of the file (after `export_data_for_mlx`):

```python
def run_mlx_training(cfg: TrainConfig) -> None:
    """Metal-only: quantizes the base model to a local 4-bit MLX copy under
    cfg.output_dir/mlx_base (this quantized-base-plus-adapter combination is
    mlx-lm's QLoRA -- there is no separate QLoRA flag), renders the
    train/val splits to mlx-lm's text format under cfg.output_dir/mlx_data
    via export_data_for_mlx, and runs mlx_lm.lora's training loop. Adapters
    are written to cfg.output_dir (adapters.safetensors + adapter_config.json)
    by mlx_lm.lora itself."""
    import types

    from mlx_lm import convert, load as mlx_load
    from mlx_lm.lora import run as mlx_lora_run

    from llm_internal.train.sft import load_split

    output_dir = Path(cfg.output_dir)
    base_dir = output_dir / "mlx_base"
    data_dir = output_dir / "mlx_data"

    if not base_dir.exists():
        convert(cfg.base_model, mlx_path=str(base_dir), quantize=True, q_bits=4)

    _, tokenizer = mlx_load(str(base_dir))
    export_data_for_mlx(cfg.data_dir, tokenizer, data_dir)

    train_examples = load_split(Path(cfg.data_dir) / "train.jsonl")
    args_dict = build_mlx_training_args(cfg, base_dir, data_dir, len(train_examples))
    mlx_lora_run(types.SimpleNamespace(**args_dict))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/train/test_sft.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add src/llm_internal/train/sft.py src/llm_internal/train/mlx_backend.py tests/train/test_sft.py
git commit -m "feat: add MLX training entrypoint and run_training backend dispatch"
```

---

### Task 5: `EvalConfig` backend field

**Files:**
- Modify: `src/llm_internal/eval/config.py`
- Modify: `configs/eval.yaml`
- Modify: `tests/eval/test_run_eval.py`
- Modify: `tests/test_pipeline_integration.py`

**Interfaces:**
- Produces: `EvalConfig.backend: str` (validated `"cuda"|"mlx"`), required field like every other `EvalConfig` field.

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_run_eval.py
import pytest

from llm_internal.eval.config import EvalConfig, load_eval_config
from llm_internal.eval.run_eval import evaluate_examples


def test_load_eval_config_reads_real_config_file():
    cfg = load_eval_config("configs/eval.yaml")

    assert isinstance(cfg, EvalConfig)
    assert 0 < cfg.tool_call_accuracy_threshold <= 1
    assert 0 < cfg.plain_chat_pass_rate_threshold <= 1
    assert cfg.backend == "cuda"


def test_eval_config_rejects_invalid_backend():
    with pytest.raises(ValueError, match="backend"):
        EvalConfig(
            model_dir="m", eval_file="e", max_new_tokens=1, min_plain_chat_chars=1,
            tool_call_accuracy_threshold=0.8, plain_chat_pass_rate_threshold=0.8,
            backend="tpu",
        )


def test_evaluate_examples_scores_mixed_batch_and_gates():
    examples = [
        {
            "category": "tool_call",
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'},
            ],
        },
        {
            "category": "plain_chat",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello there"},
            ],
        },
    ]
    predictions = [
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
        "Hello! How can I help you today?",
    ]
    cfg = EvalConfig(
        model_dir="unused",
        eval_file="unused",
        max_new_tokens=256,
        min_plain_chat_chars=5,
        tool_call_accuracy_threshold=0.8,
        plain_chat_pass_rate_threshold=0.8,
        backend="cuda",
    )

    report = evaluate_examples(examples, predictions, cfg)

    assert report.tool_call_accuracy == 1.0
    assert report.plain_chat_pass_rate == 1.0
    assert report.passed is True


def test_evaluate_examples_requires_matching_lengths():
    with pytest.raises(ValueError):
        evaluate_examples([{"category": "plain_chat", "messages": []}], [], EvalConfig(
            model_dir="unused", eval_file="unused", max_new_tokens=1,
            min_plain_chat_chars=1, tool_call_accuracy_threshold=0.8,
            plain_chat_pass_rate_threshold=0.8, backend="cuda",
        ))
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/eval/test_run_eval.py -v`
Expected: FAIL — `EvalConfig` doesn't accept/validate `backend` yet

- [ ] **Step 3: Modify `configs/eval.yaml`**

```yaml
model_dir: checkpoints
eval_file: data/processed/eval.jsonl
max_new_tokens: 512
min_plain_chat_chars: 5
tool_call_accuracy_threshold: 0.8
plain_chat_pass_rate_threshold: 0.8
backend: cuda
```

- [ ] **Step 4: Modify `src/llm_internal/eval/config.py`**

```python
"""Config for the held-out eval gate."""
from __future__ import annotations

import dataclasses
from pathlib import Path

from llm_internal.config_utils import load_yaml_dataclass


@dataclasses.dataclass
class EvalConfig:
    model_dir: str
    eval_file: str
    max_new_tokens: int
    min_plain_chat_chars: int
    tool_call_accuracy_threshold: float
    plain_chat_pass_rate_threshold: float
    backend: str

    def __post_init__(self) -> None:
        if self.backend not in ("cuda", "mlx"):
            raise ValueError(f"backend must be 'cuda' or 'mlx', got {self.backend!r}")


def load_eval_config(path: str | Path) -> EvalConfig:
    return load_yaml_dataclass(path, EvalConfig)
```

- [ ] **Step 5: Modify `tests/test_pipeline_integration.py`**

```python
# tests/test_pipeline_integration.py
from pathlib import Path

from llm_internal.data.transform import format_example
from llm_internal.data.prepare import prepare_dataset
from llm_internal.eval.config import EvalConfig
from llm_internal.eval.run_eval import evaluate_examples
from llm_internal.export.to_gguf import render_modelfile, write_modelfile
from llm_internal.train.sft import load_split


def _fixture_raw_examples():
    return [
        {
            "id": "tc-1",
            "conversations": [
                {"from": "system", "value": "You are a function calling AI model.\n<tools>[]</tools>"},
                {"from": "human", "value": "What's the weather in Paris?"},
                {"from": "gpt", "value": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'},
            ],
        },
        {
            "id": "pc-1",
            "conversations": [
                {"from": "system", "value": "You are a helpful assistant."},
                {"from": "human", "value": "Hi there"},
                {"from": "gpt", "value": "Hello! How can I help you today?"},
            ],
        },
    ] * 10  # 20 raw examples, enough for a non-degenerate 80/10/10 split


def test_full_offline_pipeline_slice_runs_end_to_end(tmp_path: Path):
    # 1. format
    formatted = [format_example(r) for r in _fixture_raw_examples()]

    # 2. split + write
    data_dir = tmp_path / "processed"
    counts = prepare_dataset(formatted, output_dir=data_dir, train_ratio=0.8, val_ratio=0.1, eval_ratio=0.1, seed=0)
    assert sum(counts.values()) == 20

    # 3. load the eval split back
    eval_examples = load_split(data_dir / "eval.jsonl")
    assert eval_examples

    # 4. pretend a model generated perfect predictions, score them
    cfg = EvalConfig(
        model_dir="unused", eval_file="unused", max_new_tokens=64,
        min_plain_chat_chars=3, tool_call_accuracy_threshold=0.8,
        plain_chat_pass_rate_threshold=0.8, backend="cuda",
    )
    predictions = []
    for ex in eval_examples:
        last = ex["messages"][-1]["content"]
        predictions.append(last)  # perfect prediction
    report = evaluate_examples(eval_examples, predictions, cfg)
    assert report.passed is True

    # 5. render + write the Modelfile a passing run would export
    modelfile_path = tmp_path / "export" / "Modelfile"
    write_modelfile(modelfile_path, render_modelfile("homemade-llm-q4_k_m.gguf"))
    assert modelfile_path.exists()
    assert "FROM ./homemade-llm-q4_k_m.gguf" in modelfile_path.read_text()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_run_eval.py tests/test_pipeline_integration.py -v`
Expected: PASS (5 tests)

- [ ] **Step 7: Commit**

```bash
git add src/llm_internal/eval/config.py configs/eval.yaml tests/eval/test_run_eval.py tests/test_pipeline_integration.py
git commit -m "feat: add backend field to EvalConfig"
```

---

### Task 6: MLX eval generation + `_load_and_generate` dispatcher

**Files:**
- Modify: `src/llm_internal/eval/run_eval.py`
- Modify: `tests/eval/test_run_eval.py`

**Interfaces:**
- Consumes: `EvalConfig.backend` (Task 5).
- Produces: `generate_predictions_mlx(examples, model_dir, cfg) -> list[str]` (Metal-only), `_load_and_generate(examples, cfg) -> list[str]` (dispatcher used by `main()`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_run_eval.py
import pytest

from llm_internal.eval.config import EvalConfig, load_eval_config
from llm_internal.eval.run_eval import _load_and_generate, evaluate_examples


def test_load_eval_config_reads_real_config_file():
    cfg = load_eval_config("configs/eval.yaml")

    assert isinstance(cfg, EvalConfig)
    assert 0 < cfg.tool_call_accuracy_threshold <= 1
    assert 0 < cfg.plain_chat_pass_rate_threshold <= 1
    assert cfg.backend == "cuda"


def test_eval_config_rejects_invalid_backend():
    with pytest.raises(ValueError, match="backend"):
        EvalConfig(
            model_dir="m", eval_file="e", max_new_tokens=1, min_plain_chat_chars=1,
            tool_call_accuracy_threshold=0.8, plain_chat_pass_rate_threshold=0.8,
            backend="tpu",
        )


def test_evaluate_examples_scores_mixed_batch_and_gates():
    examples = [
        {
            "category": "tool_call",
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'},
            ],
        },
        {
            "category": "plain_chat",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello there"},
            ],
        },
    ]
    predictions = [
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
        "Hello! How can I help you today?",
    ]
    cfg = EvalConfig(
        model_dir="unused",
        eval_file="unused",
        max_new_tokens=256,
        min_plain_chat_chars=5,
        tool_call_accuracy_threshold=0.8,
        plain_chat_pass_rate_threshold=0.8,
        backend="cuda",
    )

    report = evaluate_examples(examples, predictions, cfg)

    assert report.tool_call_accuracy == 1.0
    assert report.plain_chat_pass_rate == 1.0
    assert report.passed is True


def test_evaluate_examples_requires_matching_lengths():
    with pytest.raises(ValueError):
        evaluate_examples([{"category": "plain_chat", "messages": []}], [], EvalConfig(
            model_dir="unused", eval_file="unused", max_new_tokens=1,
            min_plain_chat_chars=1, tool_call_accuracy_threshold=0.8,
            plain_chat_pass_rate_threshold=0.8, backend="cuda",
        ))


def test_load_and_generate_dispatches_to_mlx(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "llm_internal.eval.run_eval.generate_predictions_mlx",
        lambda examples, model_dir, cfg: calls.append(("mlx", model_dir)) or ["pred"],
    )
    cfg = EvalConfig(
        model_dir="m", eval_file="unused", max_new_tokens=1, min_plain_chat_chars=1,
        tool_call_accuracy_threshold=0.8, plain_chat_pass_rate_threshold=0.8, backend="mlx",
    )

    result = _load_and_generate([{"messages": []}], cfg)

    assert result == ["pred"]
    assert calls == [("mlx", "m")]


def test_load_and_generate_dispatches_to_cuda(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **k: "tok"))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", classmethod(lambda cls, *a, **k: "model"))
    calls = []
    monkeypatch.setattr(
        "llm_internal.eval.run_eval.generate_predictions",
        lambda examples, model, tokenizer, cfg: calls.append((model, tokenizer)) or ["pred"],
    )
    cfg = EvalConfig(
        model_dir="m", eval_file="unused", max_new_tokens=1, min_plain_chat_chars=1,
        tool_call_accuracy_threshold=0.8, plain_chat_pass_rate_threshold=0.8, backend="cuda",
    )

    result = _load_and_generate([{"messages": []}], cfg)

    assert result == ["pred"]
    assert calls == [("model", "tok")]
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `uv run pytest tests/eval/test_run_eval.py -v`
Expected: FAIL — `_load_and_generate`/`generate_predictions_mlx` don't exist yet

- [ ] **Step 3: Modify `src/llm_internal/eval/run_eval.py`**

```python
"""Held-out eval gate. `evaluate_examples` is pure (takes pre-generated
predictions); `generate_predictions` (CUDA/transformers) and
`generate_predictions_mlx` (MLX) each require the trained model loaded on
real hardware. `_load_and_generate` dispatches between them based on
EvalConfig.backend."""
from __future__ import annotations

import sys

from llm_internal.eval.config import EvalConfig, load_eval_config
from llm_internal.eval.scoring import Report, aggregate_results, score_plain_chat_example, score_tool_call_example
from llm_internal.train.sft import load_split


def evaluate_examples(examples: list[dict], predictions: list[str], cfg: EvalConfig) -> Report:
    if len(examples) != len(predictions):
        raise ValueError(f"examples ({len(examples)}) and predictions ({len(predictions)}) length mismatch")

    per_example_results = []
    for example, predicted_text in zip(examples, predictions):
        category = example["category"]
        if category == "tool_call":
            scored = score_tool_call_example(example["messages"], predicted_text)
            per_example_results.append({
                "category": "tool_call",
                "structural_match": scored["structural_match"],
                "plain_chat_pass": None,
            })
        else:
            passed = score_plain_chat_example(predicted_text, cfg.min_plain_chat_chars)
            per_example_results.append({
                "category": "plain_chat",
                "structural_match": None,
                "plain_chat_pass": passed,
            })

    return aggregate_results(
        per_example_results,
        tool_call_accuracy_threshold=cfg.tool_call_accuracy_threshold,
        plain_chat_pass_rate_threshold=cfg.plain_chat_pass_rate_threshold,
    )


def generate_predictions(examples: list[dict], model, tokenizer, cfg: EvalConfig) -> list[str]:
    """GPU-only: for each example, render every message except the final
    assistant turn through the chat template with a generation prompt, then
    greedily generate the model's reply. enable_thinking=False matches the
    training distribution (see train/sft.py build_training_text)."""
    predictions = []
    for example in examples:
        prompt_messages = example["messages"][:-1]
        text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        predictions.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return predictions


def generate_predictions_mlx(examples: list[dict], model_dir: str, cfg: EvalConfig) -> list[str]:
    """Metal-only: mirrors generate_predictions but loads/generates via
    mlx_lm. enable_thinking=False is passed explicitly to the tokenizer's
    chat template -- unlike mlx-lm's training dataset path, the inference
    path does support this override."""
    from mlx_lm import generate as mlx_generate, load as mlx_load

    model, tokenizer = mlx_load(model_dir)
    predictions = []
    for example in examples:
        prompt_messages = example["messages"][:-1]
        text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        predictions.append(mlx_generate(model, tokenizer, prompt=text, max_tokens=cfg.max_new_tokens))
    return predictions


def _load_and_generate(examples: list[dict], cfg: EvalConfig) -> list[str]:
    """Dispatches to the CUDA (transformers) or MLX prediction path based
    on cfg.backend."""
    if cfg.backend == "mlx":
        return generate_predictions_mlx(examples, cfg.model_dir, cfg)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_dir, device_map="auto")
    return generate_predictions(examples, model, tokenizer, cfg)


def main() -> None:
    cfg = load_eval_config("configs/eval.yaml")
    examples = load_split(cfg.eval_file)

    predictions = _load_and_generate(examples, cfg)
    report = evaluate_examples(examples, predictions, cfg)

    print(f"tool_call_accuracy={report.tool_call_accuracy:.3f} "
          f"plain_chat_pass_rate={report.plain_chat_pass_rate:.3f} passed={report.passed}")
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_run_eval.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/llm_internal/eval/run_eval.py tests/eval/test_run_eval.py
git commit -m "feat: add MLX eval generation and backend dispatch"
```

---

### Task 7: `ExportConfig`

**Files:**
- Create: `src/llm_internal/export/config.py`
- Create: `configs/export.yaml`
- Test: `tests/export/test_config.py`

**Interfaces:**
- Consumes: `load_yaml_dataclass` (`config_utils.py`, existing).
- Produces: `ExportConfig` dataclass (`backend, model_dir, output_dir, quant`) + `load_export_config(path) -> ExportConfig`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/export/test_config.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/export/test_config.py -v`
Expected: FAIL / ImportError

- [ ] **Step 3: Write `configs/export.yaml`**

```yaml
backend: cuda
model_dir: checkpoints
output_dir: export
quant: q4_k_m
```

- [ ] **Step 4: Write `src/llm_internal/export/config.py`**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/export/test_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add src/llm_internal/export/config.py configs/export.yaml tests/export/test_config.py
git commit -m "feat: add ExportConfig with backend field"
```

---

### Task 8: MLX export (fuse + quantize + README)

**Files:**
- Create: `src/llm_internal/export/to_mlx.py`
- Test: `tests/export/test_to_mlx.py`

**Interfaces:**
- Produces: `render_mlx_readme(output_dir, system_prompt=DEFAULT_SYSTEM_PROMPT) -> str` (pure), `write_readme(path, content) -> None` (pure), `fuse_and_quantize_mlx(model_dir, output_dir, q_bits=4) -> Path` (Metal-only).

- [ ] **Step 1: Write the failing tests**

```python
# tests/export/test_to_mlx.py
from pathlib import Path

from llm_internal.export.to_mlx import render_mlx_readme, write_readme


def test_render_mlx_readme_references_output_dir_and_enable_thinking_false():
    content = render_mlx_readme("export")

    assert "mlx_lm.generate --model export" in content
    assert '"enable_thinking": false' in content
    assert "mlx_lm.server --model export" in content


def test_write_readme_creates_file_with_content(tmp_path: Path):
    path = tmp_path / "README.md"

    write_readme(path, "# hi\n")

    assert path.read_text() == "# hi\n"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/export/test_to_mlx.py -v`
Expected: FAIL / ImportError

- [ ] **Step 3: Write `src/llm_internal/export/to_mlx.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/export/test_to_mlx.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/llm_internal/export/to_mlx.py tests/export/test_to_mlx.py
git commit -m "feat: add MLX fuse/quantize export and README rendering"
```

---

### Task 9: `run_export` dispatcher

**Files:**
- Create: `src/llm_internal/export/run_export.py`
- Test: `tests/export/test_run_export.py`

**Interfaces:**
- Consumes: `ExportConfig`/`load_export_config` (Task 7); `fuse_and_quantize_mlx`/`render_mlx_readme`/`write_readme` (Task 8); `merge_and_quantize`/`render_modelfile`/`write_modelfile` (`export/to_gguf.py`, existing, unchanged).
- Produces: `run_export(cfg: ExportConfig) -> Path`, `main() -> None` CLI entrypoint.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/export/test_run_export.py -v`
Expected: FAIL / ImportError

- [ ] **Step 3: Write `src/llm_internal/export/run_export.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/export/test_run_export.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/llm_internal/export/run_export.py tests/export/test_run_export.py
git commit -m "feat: add config-driven export dispatcher"
```

---

### Task 10: Mac run script + README

**Files:**
- Create: `scripts/run_on_mac_mlx.sh`
- Modify: `README.md`

**Interfaces:**
- Consumes: `configs/train.yaml`, `configs/eval.yaml`, `configs/export.yaml` (all with `backend: mlx` set), and `llm_internal.data.prepare`/`train.sft`/`eval.run_eval`/`export.run_export`'s `main()` entrypoints.
- Produces: a documented, ordered run procedure for Apple Silicon, and a top-level README covering both backends.

- [ ] **Step 1: Write `scripts/run_on_mac_mlx.sh`**

```bash
#!/usr/bin/env bash
# Run the full fine-tuning pipeline on Apple Silicon via MLX.
# Prerequisites: macOS on Apple Silicon, Python 3.10+, this repo cloned,
# and `uv` installed (curl -LsSf https://astral.sh/uv/install.sh | sh).
# Requires configs/train.yaml, configs/eval.yaml, and configs/export.yaml
# to have `backend: mlx` set.
#
# Usage: ./scripts/run_on_mac_mlx.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[1/5] Installing dependencies (incl. mlx-lm)..."
uv sync --extra dev --extra mlx

echo "[2/5] Preparing dataset (downloads NousResearch/hermes-function-calling-v1)..."
uv run python -m llm_internal.data.prepare

echo "[3/5] Running MLX QLoRA fine-tune..."
uv run python -m llm_internal.train.sft

echo "[4/5] Running held-out eval gate..."
if ! uv run python -m llm_internal.eval.run_eval; then
    echo "Eval gate failed -- checkpoint not exported. Inspect metrics above, adjust configs/train.yaml, and re-run training." >&2
    exit 1
fi

echo "[5/5] Exporting fused + quantized MLX model..."
uv run python -m llm_internal.export.run_export

echo "Done. See export/README.md for how to run it (mlx_lm.generate / mlx_lm.server)."
```

- [ ] **Step 2: Make it executable and syntax-check it**

Run: `chmod +x scripts/run_on_mac_mlx.sh && bash -n scripts/run_on_mac_mlx.sh`
Expected: exits 0 (no syntax errors)

- [ ] **Step 3: Modify `README.md`**

```markdown
# llm_internal — Homemade LLM

QLoRA fine-tune of `Qwen/Qwen3-1.7B` for reliable tool calling, trained on
`NousResearch/hermes-function-calling-v1`, deployed locally via
Ollama/llama.cpp (CUDA backend) or `mlx_lm` (MLX backend).

Design: `docs/superpowers/specs/2026-08-07-homemade-llm-design.md`,
`docs/superpowers/specs/2026-08-07-mlx-support-design.md`
Plan: `docs/superpowers/plans/2026-08-07-homemade-llm.md`,
`docs/superpowers/plans/2026-08-07-mlx-support.md`

## Pipeline

1. **Prepare data** — download + format + split (`llm_internal.data.prepare`), backend-agnostic
2. **Train** — QLoRA SFT (`llm_internal.train.sft`), dispatches on `configs/train.yaml`'s `backend`
3. **Eval** — held-out gate on tool-call structural accuracy + plain-chat pass rate (`llm_internal.eval.run_eval`), dispatches on `configs/eval.yaml`'s `backend`
4. **Export** — merge/fuse + quantize (`llm_internal.export.run_export`), dispatches on `configs/export.yaml`'s `backend`

## Backends

| | `backend: cuda` (default) | `backend: mlx` |
|---|---|---|
| Hardware | Rented CUDA GPU | Apple Silicon Mac |
| Training | Unsloth `FastLanguageModel` + TRL `SFTTrainer` | `mlx_lm.lora` (QLoRA on a locally quantized base) |
| Eval generation | `transformers` | `mlx_lm.generate` |
| Export output | GGUF (`q4_k_m`) + Ollama `Modelfile` | Fused + quantized MLX weights dir + `README.md` |
| Run script | `./scripts/run_on_runpod.sh` | `./scripts/run_on_mac_mlx.sh` |
| Install | `uv sync --extra dev` | `uv sync --extra dev --extra mlx` |

Switching backends is a config edit (`backend: cuda` / `backend: mlx` in
`configs/train.yaml`, `configs/eval.yaml`, `configs/export.yaml`), not a
code change. `mlx-lm` cannot produce GGUF for Qwen3, so the MLX export
output is a genuinely different artifact (see
`docs/superpowers/specs/2026-08-07-mlx-support-design.md`).

Step 1, and all pure logic (`data/transform.py`, `eval/scoring.py`,
`train/mlx_backend.py`'s config/data helpers, `export/to_gguf.py` and
`export/to_mlx.py`'s rendering functions), run and are unit-tested locally
with no GPU and no `mlx` install.

## Local development

```bash
uv sync --extra dev
uv run pytest
```

## Config

- `configs/data.yaml` — dataset source, revision pin, split ratios
- `configs/train.yaml` — base model, LoRA hyperparameters, training schedule, `backend`
- `configs/eval.yaml` — eval gate thresholds, `backend`
- `configs/export.yaml` — export model/output dirs, quant level, `backend`

## After training (CUDA backend)

Copy `export/*.gguf` and `export/Modelfile` off the rented pod, then locally:

```bash
ollama create homemade-llm -f Modelfile
ollama run homemade-llm
```

## After training (MLX backend)

`export/` holds a ready-to-run MLX model directory. See `export/README.md`
(generated at export time), or directly:

```bash
mlx_lm.generate --model export --chat-template-args '{"enable_thinking": false}' --prompt "..."
mlx_lm.server --model export
```
```

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest -v`
Expected: PASS — every test from this plan's Tasks 1–9 plus the unchanged base-plan tests (no regressions)

- [ ] **Step 5: Commit**

```bash
git add scripts/run_on_mac_mlx.sh README.md
git commit -m "docs: add MLX run script and update README for dual-backend support"
```
