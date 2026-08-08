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
| Run script | `./scripts/run_on_runpod.sh` (rented GPU) or `./scripts/run_on_kaggle.sh` (free Kaggle T4) | `./scripts/run_on_mac_mlx.sh` |
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

## Running on Kaggle (free GPU)

`backend: cuda` also runs on Kaggle's free T4 quota (~30 GPU-hours/week,
~9-12h/session cap) via `./scripts/run_on_kaggle.sh` — no account billing,
no config changes vs. `run_on_runpod.sh`. Training auto-resumes from the
latest checkpoint, so a run that outlives one session continues in the
next: `Save Version` (with "Always save output") to persist `checkpoints/`
as notebook output, turn that output into a Kaggle Dataset, attach it as
input to the next session, and pass its path as `RESUME_FROM`. See the
script header for the exact cell commands.

For a fully local, notebook-free workflow, `./scripts/run_on_kaggle_api.sh`
drives the same Kaggle run through the `kaggle` API/CLI: it pushes a
script kernel (`scripts/kaggle_api/`), polls until the run finishes,
downloads output to `./kaggle_output/`, and — if training didn't finish —
publishes `kaggle_output/llm_internal/checkpoints/` as a Kaggle Dataset and
wires it in as the next run's resume source automatically. Re-run it
(e.g. weekly, as the GPU quota resets) until `export/*.gguf` appears.

One-time setup:

1. `pip install kaggle` (kaggle CLI 2.x; also needs `jq`, used to patch
   the kernel/dataset metadata).
2. Kaggle account → Settings → API → "Create New Token" → save the token
   string to `~/.kaggle/access_token` (`chmod 600`). kaggle CLI 2.x no
   longer reads the old `~/.kaggle/kaggle.json` username+key format;
   `export KAGGLE_API_TOKEN=<token>` also works.
3. In `scripts/kaggle_api/kernel-metadata.json`, set `"id"` to
   `<your-kaggle-username>/homemade-llm-training`.
4. In `scripts/kaggle_api/run_kernel.py`, set `REPO_URL` to this repo's git
   remote — it must be `git clone`-able from Kaggle with internet enabled
   (a public URL, or an `https://` URL with an embedded token for a
   private repo).

Then run:

```bash
KAGGLE_USERNAME=<your-kaggle-username> ./scripts/run_on_kaggle_api.sh
```

Each invocation pushes the kernel, blocks until it completes (or fails),
and syncs `./kaggle_output/`. If `kaggle_output/llm_internal/export/*.gguf`
isn't there yet, just re-run the same command later — it resumes from
`kaggle_output/llm_internal/checkpoints/` automatically.

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
