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
