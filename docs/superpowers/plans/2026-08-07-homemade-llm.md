# Homemade LLM (Qwen3-1.7B QLoRA Tool-Use Fine-Tune) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full pipeline (data prep → QLoRA SFT → held-out eval gate → GGUF export) to fine-tune `Qwen/Qwen3-1.7B` for reliable tool calling, deployable locally via Ollama/llama.cpp.

**Architecture:** A `src/llm_internal` package with one focused module per pipeline stage (`data`, `train`, `eval`, `export`), each exposing pure/testable transform functions plus a thin CLI `main()` that wires them to real I/O (HF Hub downloads, GPU model load). Pure logic (chat formatting, splitting, scoring, Modelfile rendering) is unit-tested locally with no GPU. GPU-only steps (actual QLoRA training run, generation, LoRA merge/quantize) are implemented for real but exercised on a rented GPU per `scripts/run_on_runpod.sh` — this workstation has no CUDA device.

**Tech Stack:** Python 3.10+, uv, PyTorch, Unsloth, TRL (`SFTTrainer`), PEFT, Transformers, Hugging Face `datasets`/`huggingface_hub`, PyYAML, pytest.

## Global Constraints

- Base model: `Qwen/Qwen3-1.7B` (confirmed via HF model card: native tool calling, `<tool_call>` tag convention, thinking/non-thinking switch).
- Dataset: `NousResearch/hermes-function-calling-v1`, pinned to revision `dae3e1d28cfbcf4b915c04ea1e072030529b4bda`. Files used: `func-calling.json`, `func-calling-singleturn.json` (the actual function-calling conversations; `json-mode-*` files are a different task and excluded — YAGNI).
- Split: 90% train / 5% val / 5% held-out eval, stratified by category (`tool_call` vs `plain_chat`), seed 42.
- Method: 4-bit QLoRA via Unsloth `FastLanguageModel` + TRL `SFTTrainer`.
- Training/eval never fabricate `enable_thinking=True` content — see Task 5 finding: Qwen3's chat template auto-inserts an empty `<think>\n\n</think>\n\n` block on the final assistant turn during full-conversation rendering (no explicit flag needed there); `enable_thinking=False` must be passed explicitly wherever `add_generation_prompt=True` is used (eval/inference), matching training distribution.
- Held-out eval gate: tool-call structural accuracy ≥ 0.8, plain-chat pass rate ≥ 0.8 (from `configs/eval.yaml`). A failing gate must exit non-zero and block export.
- Export target: GGUF via Unsloth `save_pretrained_gguf` (start quant: `q4_k_m`) + generated `Modelfile` for `ollama create`.
- No project-wide test/lint/format commands are invented beyond `pytest` — run only the tests named in each task.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/llm_internal/__init__.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`

**Interfaces:**
- Produces: importable package `llm_internal` (version string `llm_internal.__version__`), a working `uv run pytest` harness all later tasks add tests to.

- [ ] **Step 1: Write `pyproject.toml`**

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

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/llm_internal"]
```

- [ ] **Step 2: Write `src/llm_internal/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Write `tests/__init__.py`**

```python
```

(empty — marks `tests` as a package so relative imports/fixtures resolve consistently)

- [ ] **Step 4: Write `.gitignore`**

```
__pycache__/
*.pyc
.venv/
.pytest_cache/
data/processed/
checkpoints/
*.gguf
.env
```

- [ ] **Step 5: Install and verify**

Run: `uv sync --extra dev`
Expected: dependency resolution succeeds and creates `.venv` (this step downloads torch/unsloth/etc.; it is the heaviest install in the project and only needs to happen once).

Run: `uv run python -c "import llm_internal; print(llm_internal.__version__)"`
Expected: prints `0.1.0`

Run: `uv run pytest --collect-only`
Expected: exits 0, "no tests ran" (no test files yet)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/llm_internal/__init__.py tests/__init__.py .gitignore
git commit -m "chore: scaffold llm_internal package"
```

---

### Task 2: Chat-template transform logic

**Files:**
- Create: `src/llm_internal/data/__init__.py`
- Create: `src/llm_internal/data/transform.py`
- Test: `tests/data/test_transform.py`

**Interfaces:**
- Produces:
  - `ROLE_MAP: dict[str, str]`
  - `format_example(raw: dict) -> dict` — returns `{"id": str | None, "messages": list[{"role": str, "content": str}], "category": "tool_call" | "plain_chat"}`
  - `stratified_split(examples: list[dict], train_ratio: float, val_ratio: float, eval_ratio: float, seed: int) -> tuple[list[dict], list[dict], list[dict]]`

- [ ] **Step 1: Write `src/llm_internal/data/__init__.py`**

```python
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/data/test_transform.py
import pytest

from llm_internal.data.transform import format_example, stratified_split


def _raw(conversations, ex_id="abc"):
    return {"id": ex_id, "conversations": conversations}


def test_format_example_maps_roles_and_flags_tool_call():
    raw = _raw([
        {"from": "system", "value": "You are a function calling AI model.\n<tools>[]</tools>"},
        {"from": "human", "value": "What's the weather?"},
        {"from": "gpt", "value": '<tool_call>\n{"name": "get_weather", "arguments": {}}\n</tool_call>'},
        {"from": "tool", "value": "<tool_response>\n{\"temp\": 20}\n</tool_response>"},
        {"from": "gpt", "value": "It's 20 degrees."},
    ])

    result = format_example(raw)

    assert result["id"] == "abc"
    assert [m["role"] for m in result["messages"]] == ["system", "user", "assistant", "tool", "assistant"]
    assert result["messages"][1]["content"] == "What's the weather?"
    assert result["category"] == "tool_call"


def test_format_example_plain_chat_has_no_tool_call_tag():
    raw = _raw([
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "human", "value": "Hi"},
        {"from": "gpt", "value": "Hello! How can I help?"},
    ])

    result = format_example(raw)

    assert result["category"] == "plain_chat"


def test_format_example_rejects_unknown_role():
    raw = _raw([{"from": "narrator", "value": "..."}])

    with pytest.raises(ValueError, match="narrator"):
        format_example(raw)


def test_stratified_split_preserves_all_examples_no_overlap():
    examples = (
        [{"id": f"tc-{i}", "category": "tool_call"} for i in range(40)]
        + [{"id": f"pc-{i}", "category": "plain_chat"} for i in range(20)]
    )

    train, val, ev = stratified_split(examples, train_ratio=0.8, val_ratio=0.1, eval_ratio=0.1, seed=42)

    assert len(train) + len(val) + len(ev) == len(examples)
    train_ids = {e["id"] for e in train}
    val_ids = {e["id"] for e in val}
    eval_ids = {e["id"] for e in ev}
    assert not (train_ids & val_ids) and not (train_ids & eval_ids) and not (val_ids & eval_ids)


def test_stratified_split_keeps_category_proportions_in_each_split():
    examples = (
        [{"id": f"tc-{i}", "category": "tool_call"} for i in range(100)]
        + [{"id": f"pc-{i}", "category": "plain_chat"} for i in range(100)]
    )

    train, val, ev = stratified_split(examples, train_ratio=0.9, val_ratio=0.05, eval_ratio=0.05, seed=42)

    for split in (train, val, ev):
        tc = sum(1 for e in split if e["category"] == "tool_call")
        pc = sum(1 for e in split if e["category"] == "plain_chat")
        assert abs(tc - pc) <= 1  # balanced input categories -> balanced output


def test_stratified_split_is_deterministic_for_a_fixed_seed():
    examples = [{"id": f"e-{i}", "category": "tool_call"} for i in range(30)]

    run_a = stratified_split(examples, 0.8, 0.1, 0.1, seed=7)
    run_b = stratified_split(examples, 0.8, 0.1, 0.1, seed=7)

    assert [e["id"] for e in run_a[0]] == [e["id"] for e in run_b[0]]


def test_stratified_split_rejects_ratios_that_dont_sum_to_one():
    with pytest.raises(ValueError):
        stratified_split([], train_ratio=0.8, val_ratio=0.1, eval_ratio=0.2, seed=1)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_transform.py -v`
Expected: FAIL / ImportError — `llm_internal.data.transform` does not exist yet

- [ ] **Step 4: Write `src/llm_internal/data/transform.py`**

```python
"""Pure transforms: raw hermes-function-calling-v1 examples -> Qwen3-ready
chat examples, and a stratified train/val/eval split."""
from __future__ import annotations

import random

ROLE_MAP = {
    "system": "system",
    "human": "user",
    "gpt": "assistant",
    "tool": "tool",
}


def format_example(raw: dict) -> dict:
    """Convert one raw hermes-function-calling-v1 example (`{"id", "conversations"}`,
    each conversation turn `{"from", "value"}`) into `{"id", "messages", "category"}`
    where `messages` is a list of `{"role", "content"}` dicts using Qwen3 chat-template
    role names, and `category` is `"tool_call"` if any assistant turn contains a
    `<tool_call>` block, else `"plain_chat"`.
    """
    messages = []
    for turn in raw["conversations"]:
        role = ROLE_MAP.get(turn["from"])
        if role is None:
            raise ValueError(f"unknown role {turn['from']!r} in example {raw.get('id')!r}")
        messages.append({"role": role, "content": turn["value"]})

    category = "tool_call" if any(
        m["role"] == "assistant" and "<tool_call>" in m["content"] for m in messages
    ) else "plain_chat"

    return {"id": raw.get("id"), "messages": messages, "category": category}


def stratified_split(
    examples: list[dict],
    train_ratio: float,
    val_ratio: float,
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split `examples` (each with a `"category"` key) into train/val/eval lists,
    preserving each category's proportions in every split. Deterministic for a
    given `seed`.
    """
    if abs((train_ratio + val_ratio + eval_ratio) - 1.0) > 1e-9:
        raise ValueError(
            f"train_ratio + val_ratio + eval_ratio must equal 1.0, got "
            f"{train_ratio} + {val_ratio} + {eval_ratio}"
        )

    rng = random.Random(seed)
    by_category: dict[str, list[dict]] = {}
    for ex in examples:
        by_category.setdefault(ex["category"], []).append(ex)

    train: list[dict] = []
    val: list[dict] = []
    ev: list[dict] = []
    for items in by_category.values():
        shuffled = items[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])
        ev.extend(shuffled[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(ev)
    return train, val, ev
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_transform.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Commit**

```bash
git add src/llm_internal/data/__init__.py src/llm_internal/data/transform.py tests/data/test_transform.py
git commit -m "feat: add chat formatting and stratified split"
```

---

### Task 3: Dataset prepare orchestrator

**Files:**
- Create: `src/llm_internal/config_utils.py`
- Create: `src/llm_internal/data/config.py`
- Create: `src/llm_internal/data/prepare.py`
- Create: `configs/data.yaml`
- Test: `tests/test_config_utils.py`
- Test: `tests/data/test_prepare.py`

**Interfaces:**
- Consumes: `format_example`, `stratified_split` from Task 2 (`llm_internal.data.transform`).
- Produces:
  - `load_yaml_dataclass(path, cls)` in `config_utils.py` (generic loader reused by Tasks 4 and 7).
  - `DataConfig` dataclass + `load_data_config(path) -> DataConfig` in `data/config.py`.
  - `write_jsonl(examples: list[dict], path: Path) -> None`
  - `prepare_dataset(raw_examples: list[dict], output_dir: Path, train_ratio: float = 0.9, val_ratio: float = 0.05, eval_ratio: float = 0.05, seed: int = 42) -> dict[str, int]` (returns `{"train": N, "val": N, "eval": N}`, writes `train.jsonl`/`val.jsonl`/`eval.jsonl` into `output_dir`)
  - `download_raw_examples(dataset_repo: str, dataset_revision: str, dataset_files: list[str]) -> list[dict]`
  - `main() -> None` CLI entrypoint

- [ ] **Step 1: Write the failing test for `config_utils`**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_config_utils.py -v`
Expected: FAIL / ImportError

- [ ] **Step 3: Write `src/llm_internal/config_utils.py`**

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_config_utils.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Write the failing tests for `data/prepare.py` and `data/config.py`**

```python
# tests/data/test_prepare.py
import json
from pathlib import Path

from llm_internal.data.config import DataConfig, load_data_config
from llm_internal.data.prepare import prepare_dataset, write_jsonl
from llm_internal.data.transform import format_example


def _raw_examples(n_tool_call: int, n_plain: int):
    examples = []
    for i in range(n_tool_call):
        examples.append({
            "id": f"tc-{i}",
            "conversations": [
                {"from": "system", "value": "sys<tools>[]</tools>"},
                {"from": "human", "value": f"query {i}"},
                {"from": "gpt", "value": '<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>'},
            ],
        })
    for i in range(n_plain):
        examples.append({
            "id": f"pc-{i}",
            "conversations": [
                {"from": "system", "value": "sys"},
                {"from": "human", "value": f"hi {i}"},
                {"from": "gpt", "value": "hello"},
            ],
        })
    return examples


def test_write_jsonl_round_trips(tmp_path: Path):
    examples = [{"a": 1}, {"a": 2}]
    path = tmp_path / "out.jsonl"

    write_jsonl(examples, path)

    lines = path.read_text().strip().splitlines()
    assert [json.loads(line) for line in lines] == examples


def test_prepare_dataset_writes_three_split_files(tmp_path: Path):
    raw = _raw_examples(n_tool_call=40, n_plain=40)
    formatted = [format_example(r) for r in raw]

    counts = prepare_dataset(formatted, output_dir=tmp_path, train_ratio=0.8, val_ratio=0.1, eval_ratio=0.1, seed=1)

    assert counts == {"train": 64, "val": 8, "eval": 8}
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "val.jsonl").exists()
    assert (tmp_path / "eval.jsonl").exists()

    train_lines = (tmp_path / "train.jsonl").read_text().strip().splitlines()
    assert len(train_lines) == 64
    first = json.loads(train_lines[0])
    assert "messages" in first and "category" in first


def test_load_data_config_reads_real_config_file():
    cfg = load_data_config("configs/data.yaml")

    assert isinstance(cfg, DataConfig)
    assert cfg.dataset_repo == "NousResearch/hermes-function-calling-v1"
    assert cfg.dataset_revision == "dae3e1d28cfbcf4b915c04ea1e072030529b4bda"
    assert cfg.dataset_files == ["func-calling.json", "func-calling-singleturn.json"]
    assert abs(cfg.train_ratio + cfg.val_ratio + cfg.eval_ratio - 1.0) < 1e-9
```

- [ ] **Step 6: Run to verify it fails**

Run: `uv run pytest tests/data/test_prepare.py -v`
Expected: FAIL / ImportError (`llm_internal.data.prepare`, `llm_internal.data.config`, and `configs/data.yaml` don't exist yet)

- [ ] **Step 7: Write `configs/data.yaml`**

```yaml
dataset_repo: NousResearch/hermes-function-calling-v1
dataset_revision: dae3e1d28cfbcf4b915c04ea1e072030529b4bda
dataset_files:
  - func-calling.json
  - func-calling-singleturn.json
output_dir: data/processed
train_ratio: 0.9
val_ratio: 0.05
eval_ratio: 0.05
seed: 42
```

- [ ] **Step 8: Write `src/llm_internal/data/config.py`**

```python
"""Config for the dataset preparation stage."""
from __future__ import annotations

import dataclasses
from pathlib import Path

from llm_internal.config_utils import load_yaml_dataclass


@dataclasses.dataclass
class DataConfig:
    dataset_repo: str
    dataset_revision: str
    dataset_files: list[str]
    output_dir: str
    train_ratio: float
    val_ratio: float
    eval_ratio: float
    seed: int


def load_data_config(path: str | Path) -> DataConfig:
    return load_yaml_dataclass(path, DataConfig)
```

- [ ] **Step 9: Write `src/llm_internal/data/prepare.py`**

```python
"""Orchestrates dataset preparation: download raw hermes-function-calling-v1
files, format them for Qwen3, split, and write train/val/eval JSONL files."""
from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from llm_internal.data.config import DataConfig, load_data_config
from llm_internal.data.transform import format_example, stratified_split


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def prepare_dataset(
    raw_examples: list[dict],
    output_dir: Path,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    eval_ratio: float = 0.05,
    seed: int = 42,
) -> dict[str, int]:
    """`raw_examples` are already-formatted examples (output of `format_example`,
    each with a `"category"` key). Splits them and writes train/val/eval.jsonl
    into `output_dir`. Returns the example count per split.
    """
    train, val, ev = stratified_split(raw_examples, train_ratio, val_ratio, eval_ratio, seed)
    output_dir = Path(output_dir)
    write_jsonl(train, output_dir / "train.jsonl")
    write_jsonl(val, output_dir / "val.jsonl")
    write_jsonl(ev, output_dir / "eval.jsonl")
    return {"train": len(train), "val": len(val), "eval": len(ev)}


def download_raw_examples(dataset_repo: str, dataset_revision: str, dataset_files: list[str]) -> list[dict]:
    """Download each `dataset_files` entry from `dataset_repo` at the pinned
    `dataset_revision` and concatenate their JSON list contents.
    """
    merged: list[dict] = []
    for filename in dataset_files:
        local_path = hf_hub_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            filename=filename,
            revision=dataset_revision,
        )
        with open(local_path, "r", encoding="utf-8") as f:
            merged.extend(json.load(f))
    return merged


def main() -> None:
    cfg: DataConfig = load_data_config("configs/data.yaml")
    raw = download_raw_examples(cfg.dataset_repo, cfg.dataset_revision, cfg.dataset_files)
    formatted = [format_example(r) for r in raw]
    counts = prepare_dataset(
        formatted,
        output_dir=Path(cfg.output_dir),
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        eval_ratio=cfg.eval_ratio,
        seed=cfg.seed,
    )
    print(f"wrote splits: {counts}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_prepare.py -v`
Expected: PASS (3 tests)

- [ ] **Step 11: Commit**

```bash
git add src/llm_internal/config_utils.py src/llm_internal/data/config.py src/llm_internal/data/prepare.py configs/data.yaml tests/test_config_utils.py tests/data/test_prepare.py
git commit -m "feat: add dataset prepare orchestrator and config loading"
```

---

### Task 4: Training config

**Files:**
- Create: `src/llm_internal/train/__init__.py`
- Create: `src/llm_internal/train/config.py`
- Create: `configs/train.yaml`
- Test: `tests/train/test_config.py`

**Interfaces:**
- Consumes: `load_yaml_dataclass` from Task 3 (`llm_internal.config_utils`).
- Produces: `TrainConfig` dataclass + `load_train_config(path) -> TrainConfig`, fields: `base_model, data_dir, output_dir, lora_r, lora_alpha, lora_dropout, target_modules, learning_rate, epochs, per_device_train_batch_size, gradient_accumulation_steps, max_seq_length, checkpoint_every_steps, enable_thinking, seed`. `enable_thinking` must be `False`; the loader raises `ValueError` otherwise (non-thinking mode is a hard project constraint — see Global Constraints).

- [ ] **Step 1: Write `src/llm_internal/train/__init__.py`**

```python
```

- [ ] **Step 2: Write the failing tests**

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
        "seed: 42\n"
    )

    with pytest.raises(ValueError, match="enable_thinking"):
        load_train_config(bad)
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/train/test_config.py -v`
Expected: FAIL / ImportError

- [ ] **Step 4: Write `configs/train.yaml`**

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
seed: 42
```

- [ ] **Step 5: Write `src/llm_internal/train/config.py`**

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
    seed: int

    def __post_init__(self) -> None:
        if self.enable_thinking is not False:
            raise ValueError(
                "enable_thinking must be false: the training dataset has no "
                "<think> content, so thinking mode must stay disabled to keep "
                "training and inference on-distribution"
            )


def load_train_config(path: str | Path) -> TrainConfig:
    return load_yaml_dataclass(path, TrainConfig)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/train/test_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 7: Commit**

```bash
git add src/llm_internal/train/__init__.py src/llm_internal/train/config.py configs/train.yaml tests/train/test_config.py
git commit -m "feat: add training config with enforced non-thinking mode"
```

---

### Task 5: Training entrypoint

**Files:**
- Create: `src/llm_internal/train/sft.py`
- Test: `tests/train/test_sft.py`

**Interfaces:**
- Consumes: `TrainConfig`/`load_train_config` from Task 4; JSONL files produced by Task 3's `prepare_dataset`.
- Produces:
  - `build_training_text(messages: list[dict], tokenizer) -> str` (pure given a tokenizer; no GPU)
  - `load_split(path: Path) -> list[dict]`
  - `build_hf_dataset(examples: list[dict], tokenizer) -> "datasets.Dataset"` (adds a `"text"` column)
  - `run_training(cfg: TrainConfig) -> None` (GPU-only: Unsloth model load, LoRA, `SFTTrainer.train()`, resumable)
  - `main() -> None` CLI entrypoint

**Note on scope:** `build_training_text`/`load_split`/`build_hf_dataset` are unit-tested here with a real Qwen3-1.7B tokenizer download (small, no GPU needed). `run_training` requires an actual CUDA GPU (bitsandbytes 4-bit + Unsloth) which this workstation does not have — it is implemented for real but only exercised via the documented smoke test on the rented GPU (see `scripts/run_on_runpod.sh` in Task 9), not executed in this local test suite.

- [ ] **Step 1: Write the failing tests**

```python
# tests/train/test_sft.py
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from llm_internal.train.sft import build_hf_dataset, build_training_text, load_split
from llm_internal.data.prepare import write_jsonl


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

    # Qwen3's template auto-inserts an empty think block on the final
    # assistant turn of a full conversation; no reasoning content leaks in.
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/train/test_sft.py -v`
Expected: FAIL / ImportError

- [ ] **Step 3: Write `src/llm_internal/train/sft.py`**

```python
"""QLoRA SFT training stage. Chat rendering/dataset assembly are pure and
tested locally; `run_training` requires a CUDA GPU and is exercised on a
rented GPU (see scripts/run_on_runpod.sh)."""
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
    """GPU-only: loads Qwen3-1.7B in 4-bit via Unsloth, attaches a LoRA
    adapter, and runs TRL's SFTTrainer. Resumes automatically from the
    latest checkpoint in `cfg.output_dir` if one exists.
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/train/test_sft.py -v`
Expected: PASS (4 tests) — first run downloads the Qwen3-1.7B tokenizer (network required, cached afterward)

- [ ] **Step 5: Commit**

```bash
git add src/llm_internal/train/sft.py tests/train/test_sft.py
git commit -m "feat: add QLoRA training entrypoint"
```

---

### Task 6: Eval scoring logic

**Files:**
- Create: `src/llm_internal/eval/__init__.py`
- Create: `src/llm_internal/eval/scoring.py`
- Test: `tests/eval/test_scoring.py`

**Interfaces:**
- Produces:
  - `parse_tool_calls(text: str) -> list[dict]` (extracts every `<tool_call>{...}</tool_call>` block, parses JSON; malformed blocks are skipped, never raise)
  - `score_tool_call_example(expected_messages: list[dict], predicted_text: str) -> dict` (returns `{"correct_name": bool, "correct_args": bool, "structural_match": bool}`, comparing against the last expected assistant turn)
  - `score_plain_chat_example(predicted_text: str, min_chars: int) -> bool`
  - `Report` dataclass (`tool_call_accuracy: float, plain_chat_pass_rate: float, passed: bool`)
  - `aggregate_results(per_example_results: list[dict], tool_call_accuracy_threshold: float, plain_chat_pass_rate_threshold: float) -> Report` — each item in `per_example_results` is `{"category": "tool_call" | "plain_chat", "structural_match": bool | None, "plain_chat_pass": bool | None}`

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_scoring.py
from llm_internal.eval.scoring import (
    Report,
    aggregate_results,
    parse_tool_calls,
    score_plain_chat_example,
    score_tool_call_example,
)


def test_parse_tool_calls_extracts_one_call():
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'

    calls = parse_tool_calls(text)

    assert calls == [{"name": "get_weather", "arguments": {"city": "Paris"}}]


def test_parse_tool_calls_extracts_multiple_calls():
    text = (
        '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\n'
        '<tool_call>\n{"name": "b", "arguments": {"x": 1}}\n</tool_call>'
    )

    calls = parse_tool_calls(text)

    assert [c["name"] for c in calls] == ["a", "b"]


def test_parse_tool_calls_skips_malformed_json_without_raising():
    text = "<tool_call>\nnot json\n</tool_call>"

    assert parse_tool_calls(text) == []


def test_parse_tool_calls_returns_empty_for_plain_text():
    assert parse_tool_calls("just a normal reply") == []


def test_score_tool_call_example_matches_name_and_arguments():
    expected_messages = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'},
    ]
    predicted = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'

    result = score_tool_call_example(expected_messages, predicted)

    assert result == {"correct_name": True, "correct_args": True, "structural_match": True}


def test_score_tool_call_example_flags_wrong_name():
    expected_messages = [
        {"role": "assistant", "content": '<tool_call>\n{"name": "get_weather", "arguments": {}}\n</tool_call>'},
    ]
    predicted = '<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>'

    result = score_tool_call_example(expected_messages, predicted)

    assert result["correct_name"] is False
    assert result["structural_match"] is False


def test_score_tool_call_example_flags_wrong_arguments():
    expected_messages = [
        {"role": "assistant", "content": '<tool_call>\n{"name": "f", "arguments": {"x": 1}}\n</tool_call>'},
    ]
    predicted = '<tool_call>\n{"name": "f", "arguments": {"x": 2}}\n</tool_call>'

    result = score_tool_call_example(expected_messages, predicted)

    assert result["correct_name"] is True
    assert result["correct_args"] is False
    assert result["structural_match"] is False


def test_score_plain_chat_example_passes_on_reasonable_length():
    assert score_plain_chat_example("Sure, here's the answer you asked for.", min_chars=5) is True


def test_score_plain_chat_example_fails_on_empty_or_too_short():
    assert score_plain_chat_example("", min_chars=5) is False
    assert score_plain_chat_example("ok", min_chars=5) is False


def test_aggregate_results_computes_rates_and_gate():
    results = [
        {"category": "tool_call", "structural_match": True, "plain_chat_pass": None},
        {"category": "tool_call", "structural_match": False, "plain_chat_pass": None},
        {"category": "tool_call", "structural_match": True, "plain_chat_pass": None},
        {"category": "tool_call", "structural_match": True, "plain_chat_pass": None},
        {"category": "plain_chat", "structural_match": None, "plain_chat_pass": True},
        {"category": "plain_chat", "structural_match": None, "plain_chat_pass": True},
    ]

    report = aggregate_results(results, tool_call_accuracy_threshold=0.7, plain_chat_pass_rate_threshold=0.8)

    assert isinstance(report, Report)
    assert report.tool_call_accuracy == 0.75
    assert report.plain_chat_pass_rate == 1.0
    assert report.passed is True


def test_aggregate_results_fails_gate_below_threshold():
    results = [
        {"category": "tool_call", "structural_match": False, "plain_chat_pass": None},
        {"category": "tool_call", "structural_match": True, "plain_chat_pass": None},
    ]

    report = aggregate_results(results, tool_call_accuracy_threshold=0.8, plain_chat_pass_rate_threshold=0.8)

    assert report.tool_call_accuracy == 0.5
    assert report.passed is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/eval/test_scoring.py -v`
Expected: FAIL / ImportError

- [ ] **Step 3: Write `src/llm_internal/eval/__init__.py`**

```python
```

- [ ] **Step 4: Write `src/llm_internal/eval/scoring.py`**

```python
"""Pure scoring logic for the held-out eval gate. No model/GPU involved --
takes already-generated prediction text."""
from __future__ import annotations

import dataclasses
import json
import re

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict]:
    """Extract every <tool_call>{json}</tool_call> block from `text`. Blocks
    that don't parse as JSON are skipped (a malformed model output is a
    scoring failure, not a crash)."""
    calls = []
    for raw in _TOOL_CALL_RE.findall(text):
        try:
            calls.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return calls


def _last_expected_tool_call(expected_messages: list[dict]) -> dict | None:
    for message in reversed(expected_messages):
        if message["role"] == "assistant" and "<tool_call>" in message["content"]:
            calls = parse_tool_calls(message["content"])
            return calls[0] if calls else None
    return None


def score_tool_call_example(expected_messages: list[dict], predicted_text: str) -> dict:
    """Compare the first tool call in `predicted_text` against the first tool
    call in the last tool-calling assistant turn of `expected_messages`."""
    expected = _last_expected_tool_call(expected_messages)
    predicted_calls = parse_tool_calls(predicted_text)
    predicted = predicted_calls[0] if predicted_calls else None

    if expected is None or predicted is None:
        return {"correct_name": False, "correct_args": False, "structural_match": False}

    correct_name = predicted.get("name") == expected.get("name")
    correct_args = predicted.get("arguments") == expected.get("arguments")
    return {
        "correct_name": correct_name,
        "correct_args": correct_args,
        "structural_match": correct_name and correct_args,
    }


def score_plain_chat_example(predicted_text: str, min_chars: int) -> bool:
    return len(predicted_text.strip()) >= min_chars


@dataclasses.dataclass
class Report:
    tool_call_accuracy: float
    plain_chat_pass_rate: float
    passed: bool


def aggregate_results(
    per_example_results: list[dict],
    tool_call_accuracy_threshold: float,
    plain_chat_pass_rate_threshold: float,
) -> Report:
    tool_call_results = [r for r in per_example_results if r["category"] == "tool_call"]
    plain_chat_results = [r for r in per_example_results if r["category"] == "plain_chat"]

    tool_call_accuracy = (
        sum(1 for r in tool_call_results if r["structural_match"]) / len(tool_call_results)
        if tool_call_results else 1.0
    )
    plain_chat_pass_rate = (
        sum(1 for r in plain_chat_results if r["plain_chat_pass"]) / len(plain_chat_results)
        if plain_chat_results else 1.0
    )

    passed = (
        tool_call_accuracy >= tool_call_accuracy_threshold
        and plain_chat_pass_rate >= plain_chat_pass_rate_threshold
    )
    return Report(
        tool_call_accuracy=tool_call_accuracy,
        plain_chat_pass_rate=plain_chat_pass_rate,
        passed=passed,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_scoring.py -v`
Expected: PASS (11 tests)

- [ ] **Step 6: Commit**

```bash
git add src/llm_internal/eval/__init__.py src/llm_internal/eval/scoring.py tests/eval/test_scoring.py
git commit -m "feat: add held-out eval scoring logic"
```

---

### Task 7: Eval config and entrypoint

**Files:**
- Create: `src/llm_internal/eval/config.py`
- Create: `src/llm_internal/eval/run_eval.py`
- Create: `configs/eval.yaml`
- Test: `tests/eval/test_run_eval.py`

**Interfaces:**
- Consumes: `load_yaml_dataclass` (Task 3), `Report`/`aggregate_results`/`score_tool_call_example`/`score_plain_chat_example` (Task 6), `load_split` (Task 5).
- Produces:
  - `EvalConfig` dataclass + `load_eval_config(path) -> EvalConfig` (fields: `model_dir, eval_file, max_new_tokens, min_plain_chat_chars, tool_call_accuracy_threshold, plain_chat_pass_rate_threshold`)
  - `evaluate_examples(examples: list[dict], predictions: list[str], cfg: EvalConfig) -> Report` (pure — takes already-generated predictions)
  - `generate_predictions(examples: list[dict], model, tokenizer, cfg: EvalConfig) -> list[str]` (GPU-only)
  - `main() -> None` CLI entrypoint; exits with status 1 if `report.passed` is `False`

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
            plain_chat_pass_rate_threshold=0.8,
        ))
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/eval/test_run_eval.py -v`
Expected: FAIL / ImportError

- [ ] **Step 3: Write `configs/eval.yaml`**

```yaml
model_dir: checkpoints
eval_file: data/processed/eval.jsonl
max_new_tokens: 512
min_plain_chat_chars: 5
tool_call_accuracy_threshold: 0.8
plain_chat_pass_rate_threshold: 0.8
```

- [ ] **Step 4: Write `src/llm_internal/eval/config.py`**

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


def load_eval_config(path: str | Path) -> EvalConfig:
    return load_yaml_dataclass(path, EvalConfig)
```

- [ ] **Step 5: Write `src/llm_internal/eval/run_eval.py`**

```python
"""Held-out eval gate. `evaluate_examples` is pure (takes pre-generated
predictions); `generate_predictions` requires the trained model on a GPU."""
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


def main() -> None:
    cfg = load_eval_config("configs/eval.yaml")
    examples = load_split(cfg.eval_file)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_dir, device_map="auto")

    predictions = generate_predictions(examples, model, tokenizer, cfg)
    report = evaluate_examples(examples, predictions, cfg)

    print(f"tool_call_accuracy={report.tool_call_accuracy:.3f} "
          f"plain_chat_pass_rate={report.plain_chat_pass_rate:.3f} passed={report.passed}")
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_run_eval.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Commit**

```bash
git add src/llm_internal/eval/config.py src/llm_internal/eval/run_eval.py configs/eval.yaml tests/eval/test_run_eval.py
git commit -m "feat: add eval entrypoint with pass/fail gate"
```

---

### Task 8: GGUF export

**Files:**
- Create: `src/llm_internal/export/__init__.py`
- Create: `src/llm_internal/export/to_gguf.py`
- Test: `tests/export/test_to_gguf.py`

**Interfaces:**
- Produces:
  - `render_modelfile(gguf_filename: str, system_prompt: str) -> str`
  - `write_modelfile(path: Path, content: str) -> None`
  - `merge_and_quantize(model_dir: str, output_dir: str, quant: str = "q4_k_m") -> Path` (GPU-only: Unsloth `save_pretrained_gguf`)
  - `main() -> None` CLI entrypoint

- [ ] **Step 1: Write the failing tests**

```python
# tests/export/test_to_gguf.py
from pathlib import Path

from llm_internal.export.to_gguf import render_modelfile, write_modelfile


def test_render_modelfile_references_gguf_file_and_system_prompt():
    content = render_modelfile("model-q4_k_m.gguf", system_prompt="You are a helpful assistant with tools.")

    assert "FROM ./model-q4_k_m.gguf" in content
    assert 'SYSTEM """You are a helpful assistant with tools."""' in content


def test_write_modelfile_creates_file_with_content(tmp_path: Path):
    path = tmp_path / "Modelfile"

    write_modelfile(path, "FROM ./x.gguf\n")

    assert path.read_text() == "FROM ./x.gguf\n"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/export/test_to_gguf.py -v`
Expected: FAIL / ImportError

- [ ] **Step 3: Write `src/llm_internal/export/__init__.py`**

```python
```

- [ ] **Step 4: Write `src/llm_internal/export/to_gguf.py`**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/export/test_to_gguf.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add src/llm_internal/export/__init__.py src/llm_internal/export/to_gguf.py tests/export/test_to_gguf.py
git commit -m "feat: add GGUF export and Ollama Modelfile generation"
```

---

### Task 9: RunPod helper script and top-level README

**Files:**
- Create: `scripts/run_on_runpod.sh`
- Create: `README.md`

**Interfaces:**
- Consumes: `configs/data.yaml`, `configs/train.yaml`, `configs/eval.yaml`, and the four `main()` entrypoints from Tasks 3, 5, 7, 8.
- Produces: a documented, ordered run procedure a human follows on a rented GPU pod.

- [ ] **Step 1: Write `scripts/run_on_runpod.sh`**

```bash
#!/usr/bin/env bash
# Run the full fine-tuning pipeline on a rented GPU pod (RunPod/Lambda).
# Prerequisites: a pod image with CUDA + Python 3.10+, this repo cloned,
# and `uv` installed (curl -LsSf https://astral.sh/uv/install.sh | sh).
#
# Usage: ./scripts/run_on_runpod.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[1/5] Installing dependencies..."
uv sync --extra dev

echo "[2/5] Preparing dataset (downloads NousResearch/hermes-function-calling-v1)..."
uv run python -m llm_internal.data.prepare

echo "[3/5] Running QLoRA SFT training (resumes automatically if checkpoints exist)..."
uv run python -m llm_internal.train.sft

echo "[4/5] Running held-out eval gate..."
if ! uv run python -m llm_internal.eval.run_eval; then
    echo "Eval gate failed -- checkpoint not exported. Inspect metrics above, adjust configs/train.yaml, and re-run training." >&2
    exit 1
fi

echo "[5/5] Exporting merged model to GGUF + Ollama Modelfile..."
uv run python -m llm_internal.export.to_gguf

echo "Done. Copy export/*.gguf and export/Modelfile to your local machine, then:"
echo "  ollama create homemade-llm -f Modelfile"
echo "  ollama run homemade-llm"
echo ""
echo "Remember to stop/terminate the pod once export/*.gguf has been copied out."
```

- [ ] **Step 2: Make it executable and syntax-check it**

Run: `chmod +x scripts/run_on_runpod.sh && bash -n scripts/run_on_runpod.sh`
Expected: exits 0 (no syntax errors)

- [ ] **Step 3: Write `README.md`**

```markdown
# llm_internal — Homemade LLM

QLoRA fine-tune of `Qwen/Qwen3-1.7B` for reliable tool calling, trained on
`NousResearch/hermes-function-calling-v1`, deployed locally via Ollama/llama.cpp.

Design: `docs/superpowers/specs/2026-08-07-homemade-llm-design.md`
Plan: `docs/superpowers/plans/2026-08-07-homemade-llm.md`

## Pipeline

1. **Prepare data** — download + format + split (`llm_internal.data.prepare`)
2. **Train** — 4-bit QLoRA SFT via Unsloth + TRL (`llm_internal.train.sft`)
3. **Eval** — held-out gate on tool-call structural accuracy + plain-chat pass rate (`llm_internal.eval.run_eval`)
4. **Export** — merge LoRA, quantize to GGUF, write Ollama `Modelfile` (`llm_internal.export.to_gguf`)

Steps 2–4 require a CUDA GPU. Run them on a rented pod via:

```bash
./scripts/run_on_runpod.sh
```

Step 1, and all pure logic (`data/transform.py`, `eval/scoring.py`,
`export/to_gguf.py`'s Modelfile rendering), run and are unit-tested locally
with no GPU.

## Local development

```bash
uv sync --extra dev
uv run pytest
```

## Config

- `configs/data.yaml` — dataset source, revision pin, split ratios
- `configs/train.yaml` — base model, LoRA hyperparameters, training schedule
- `configs/eval.yaml` — eval gate thresholds

## After training

Copy `export/*.gguf` and `export/Modelfile` off the rented pod, then locally:

```bash
ollama create homemade-llm -f Modelfile
ollama run homemade-llm
```
```

- [ ] **Step 4: Commit**

```bash
git add scripts/run_on_runpod.sh README.md
git commit -m "docs: add RunPod run script and top-level README"
```

---

### Task 10: End-to-end offline integration test

**Files:**
- Create: `tests/test_pipeline_integration.py`

**Interfaces:**
- Consumes every pure function from Tasks 2, 3, 6, 8: `format_example`, `prepare_dataset`, `load_split` (Task 5), `evaluate_examples`, `render_modelfile`.
- Produces: one test proving the entire GPU-free slice of the pipeline is correctly wired end-to-end.

- [ ] **Step 1: Write the failing test**

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
        plain_chat_pass_rate_threshold=0.8,
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

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_pipeline_integration.py -v`
Expected: FAIL only if any earlier task's module is missing/broken (by this point in the plan, all imports should already exist — this step should mostly confirm wiring, not new modules)

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -v`
Expected: PASS — every test from Tasks 1–10 (config utils, transform, prepare, train config, sft chat rendering, scoring, eval, export, integration)

- [ ] **Step 4: Commit**

```bash
git add tests/test_pipeline_integration.py
git commit -m "test: add end-to-end offline pipeline integration test"
```
