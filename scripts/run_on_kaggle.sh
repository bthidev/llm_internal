#!/usr/bin/env bash
# Run the full fine-tuning pipeline (CUDA backend) in a Kaggle Notebook,
# using Kaggle's free T4 GPU quota.
#
# Notebook prerequisites:
#   - Settings > Accelerator: GPU T4 x2
#   - Settings > Internet: on (needed to pull the base model, dataset, uv)
#   - This repo cloned into the notebook, e.g. in the first cell:
#       !git clone <repo-url> /kaggle/working/llm_internal
#
# Kaggle kernels do not provide Docker/Podman. The code-correctness benchmark
# therefore fails closed with `sandbox_unavailable` on Kaggle; this is an
# infrastructure limitation, not a model-code failure. Do NOT enable the
# unsafe subprocess backend merely to make this metric run: generated model
# output is arbitrary code. Run code-correctness evaluation later on a host
# with Docker/Podman and a pre-pulled python:3.11-alpine image.
#
# Kaggle's free quota is ~30 GPU-hours/week with a hard ~9-12h session cap,
# so a full run often spans multiple sessions/weeks. This script and the
# training entrypoint both resume automatically:
#   - Training (src/llm_internal/train/sft.py) resumes from the latest
#     checkpoint-* subdir under checkpoints/ if one exists, and simply
#     continues toward the full epoch count -- if a session times out
#     mid-training, the next session's run just picks up where it left off.
#   - Kaggle sessions are ephemeral: without an explicit "Save Version"
#     (Save & Run All, with "Always save output" on), everything under
#     /kaggle/working is lost when the session ends. To carry checkpoints/
#     into the next session: Save Version -> create/update a Kaggle Dataset
#     from that version's Output -> attach the dataset as input to the next
#     notebook session -> pass its checkpoints/ path as RESUME_FROM below.
#
# Usage (from a notebook cell):
#   %cd /kaggle/working/llm_internal
#   !RESUME_FROM=/kaggle/input/homemade-llm-checkpoints/checkpoints \
#       bash scripts/run_on_kaggle.sh
# (omit RESUME_FROM for the first session)
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[0/6] Seeding checkpoints from a prior session (if RESUME_FROM is set)..."
if [ -n "${RESUME_FROM:-}" ] && [ -d "$RESUME_FROM" ]; then
    mkdir -p checkpoints
    cp -r "$RESUME_FROM"/. checkpoints/
    echo "Resumed from $RESUME_FROM"
else
    echo "No RESUME_FROM set (or path missing) -- starting fresh."
fi

echo "[1/6] Installing dependencies..."
pip install -q uv
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
echo "NOTE: Kaggle has no Docker/Podman; code_correctness results will report sandbox_unavailable and are not a valid model-quality signal here." >&2
if ! uv run python -m llm_internal.eval.compare_models configs/benchmark_eval_base.yaml configs/benchmark_eval.yaml; then
    echo "Benchmark comparison failed (fine-tuned gate failure or regression vs. base) -- checkpoint not exported." >&2
    echo "See comparison_report.json and the table above, adjust configs/train.yaml, and re-run training." >&2
    exit 1
fi

echo "[6/6] Exporting merged model to GGUF + Ollama Modelfile..."
uv run python -m llm_internal.export.to_gguf

echo "Done. checkpoints/, export/, and comparison_report.json are under"
echo "/kaggle/working/llm_internal."
echo "Now: Kaggle menu -> Save Version -> Save & Run All (Commit), with"
echo "'Always save output' checked, so checkpoints/ and export/ persist as"
echo "notebook output. Download export/*.gguf and export/Modelfile from the"
echo "Output tab, then locally:"
echo "  ollama create homemade-llm -f Modelfile"
echo "  ollama run homemade-llm"
