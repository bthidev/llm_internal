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
