#!/usr/bin/env bash
# Run the full fine-tuning pipeline on a rented GPU pod (RunPod/Lambda).
# Prerequisites: a pod image with CUDA + Python 3.10+, this repo cloned,
# and `uv` installed (curl -LsSf https://astral.sh/uv/install.sh | sh).
#
# Usage: ./scripts/run_on_runpod.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[1/6] Installing dependencies..."
uv sync --extra dev

echo "[2/6] Preparing dataset (downloads NousResearch/hermes-function-calling-v1)..."
uv run python -m llm_internal.data.prepare

echo "[3/6] Running QLoRA SFT training (resumes automatically if checkpoints exist)..."
uv run python -m llm_internal.train.sft

echo "[4/6] Running held-out eval gate..."
if ! uv run python -m llm_internal.eval.run_eval; then
    echo "Eval gate failed -- checkpoint not exported. Inspect metrics above, adjust configs/train.yaml, and re-run training." >&2
    exit 1
fi

echo "[5/6] Comparing fine-tuned model against the original base model on the independent benchmark..."
if ! uv run python -m llm_internal.eval.compare_models configs/benchmark_eval_base.yaml configs/benchmark_eval.yaml; then
    echo "Benchmark comparison failed (fine-tuned gate failure or regression vs. base) -- checkpoint not exported." >&2
    echo "See comparison_report.json and the table above, adjust configs/train.yaml, and re-run training." >&2
    exit 1
fi

echo "[6/6] Exporting merged model to GGUF + Ollama Modelfile..."
uv run python -m llm_internal.export.to_gguf

echo "Done. Copy export/*.gguf and export/Modelfile to your local machine, then:"
echo "  ollama create homemade-llm -f Modelfile"
echo "  ollama run homemade-llm"
echo ""
echo "Remember to stop/terminate the pod once export/*.gguf has been copied out."
