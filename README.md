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
