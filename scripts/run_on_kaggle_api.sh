#!/usr/bin/env bash
# Drive the Kaggle free-GPU training pipeline entirely from your local
# machine via the Kaggle API/CLI -- no manual notebook UI interaction.
#
# One-time setup:
#   1. pip install kaggle  (kaggle CLI 2.x)
#   2. Kaggle account -> Settings -> API -> Create New Token, then save
#      the token string to ~/.kaggle/access_token (chmod 600). The old
#      username+key ~/.kaggle/kaggle.json format is no longer read by
#      kaggle CLI 2.x; alternatively export KAGGLE_API_TOKEN=<token>.
#   3. Edit scripts/kaggle_api/kernel-metadata.json: set "id" to
#      "<your-kaggle-username>/homemade-llm-training"
#   4. Edit REPO_URL in scripts/kaggle_api/run_kernel.py to point at this
#      repo's git remote (must be reachable with internet enabled --
#      public repo, or an https URL embedding a token for a private one)
#
# Usage:
#   KAGGLE_USERNAME=<you> ./scripts/run_on_kaggle_api.sh
#
# Each invocation: pushes the kernel, polls until the run finishes, and
# downloads its output to ./kaggle_output/. If that output contains
# checkpoints/ with checkpoint-* subdirs, it's published (created or
# versioned) as a Kaggle Dataset "<KAGGLE_USERNAME>/homemade-llm-checkpoints"
# and wired into kernel-metadata.json's dataset_sources, so the *next*
# invocation resumes training automatically. Kaggle's free quota is
# ~30 GPU-hours/week with a session cap well under a full 3-epoch run, so
# re-run this script periodically (e.g. weekly, as the quota resets) until
# the eval gate passes and export/*.gguf shows up under
# kaggle_output/llm_internal/export/.
set -euo pipefail

: "${KAGGLE_USERNAME:?Set KAGGLE_USERNAME to your Kaggle username}"

cd "$(dirname "$0")/.."
KDIR="scripts/kaggle_api"
OUT_DIR="kaggle_output"
CHECKPOINT_DATASET_ID="$KAGGLE_USERNAME/homemade-llm-checkpoints"
# Kernel output includes the full /kaggle/working tree, i.e. the ~5GB+
# uv venv (torch/CUDA libs) alongside the files we actually want.
# kaggle kernels output has no server-side exclude, only an include
# pattern -- so always scope downloads to these paths, or risk
# hammering ListKernelSessionOutput into a rate-limit cooldown.
OUT_FILE_PATTERN='.*llm_internal/checkpoints/.*|.*llm_internal/export/.*|.*__results__\.html'

command -v kaggle >/dev/null || { echo "kaggle CLI not found -- pip install kaggle" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq not found -- required to patch kernel-metadata.json" >&2; exit 1; }

KERNEL_ID="$(jq -r .id "$KDIR/kernel-metadata.json")"
case "$KERNEL_ID" in
    REPLACE_WITH_KAGGLE_USERNAME/*)
        echo "Set \"id\" in $KDIR/kernel-metadata.json to <your-kaggle-username>/homemade-llm-training first." >&2
        exit 1
        ;;
esac

echo "[1/4] Checking for checkpoints from a previous run..."
if compgen -G "$OUT_DIR/llm_internal/checkpoints/checkpoint-*" >/dev/null 2>&1; then
    echo "Found prior checkpoints -- publishing as Kaggle Dataset $CHECKPOINT_DATASET_ID..."
    DS_DIR="$(mktemp -d)"
    cp -r "$OUT_DIR/llm_internal/checkpoints" "$DS_DIR/checkpoints"
    kaggle datasets init -p "$DS_DIR" >/dev/null
    jq --arg id "$CHECKPOINT_DATASET_ID" --arg title "homemade-llm-checkpoints" \
        '.id = $id | .title = $title' "$DS_DIR/dataset-metadata.json" > "$DS_DIR/dataset-metadata.json.tmp"
    mv "$DS_DIR/dataset-metadata.json.tmp" "$DS_DIR/dataset-metadata.json"

    if kaggle datasets status "$CHECKPOINT_DATASET_ID" >/dev/null 2>&1; then
        kaggle datasets version -p "$DS_DIR" -r zip -m "checkpoint update $(date -u +%FT%TZ)"
    else
        kaggle datasets create -p "$DS_DIR" -r zip
    fi
    rm -rf "$DS_DIR"

    jq --arg ds "$CHECKPOINT_DATASET_ID" '.dataset_sources = [$ds]' \
        "$KDIR/kernel-metadata.json" > "$KDIR/kernel-metadata.json.tmp"
    mv "$KDIR/kernel-metadata.json.tmp" "$KDIR/kernel-metadata.json"
else
    echo "No prior checkpoints found in $OUT_DIR -- starting fresh."
fi

echo "[2/4] Pushing and starting the kernel run..."
kaggle kernels push -p "$KDIR" --accelerator NvidiaTeslaT4

echo "[3/4] Polling kernel status (a full session can take hours)..."
while true; do
    line="$(kaggle kernels status "$KERNEL_ID" 2>&1)"
    status="$(echo "$line" | grep -oE '"[A-Za-z_.]+"' | tr -d '"' | sed 's/.*\.//' | tail -1)"
    echo "  $line"
    case "$status" in
        COMPLETE) break ;;
        FAILED|CANCELLED|ERROR)
            echo "Kernel run did not complete. Downloading output for inspection..." >&2
            rm -rf "$OUT_DIR"
            kaggle kernels output "$KERNEL_ID" -p "$OUT_DIR" --file-pattern "$OUT_FILE_PATTERN" || true
            echo "See $OUT_DIR/ (script.log has the traceback) and:" >&2
            echo "  kaggle kernels logs $KERNEL_ID" >&2
            exit 1
            ;;
    esac
    sleep 60
done

echo "[4/4] Downloading kernel output to $OUT_DIR/ ..."
rm -rf "$OUT_DIR"
kaggle kernels output "$KERNEL_ID" -p "$OUT_DIR" --file-pattern "$OUT_FILE_PATTERN"

if compgen -G "$OUT_DIR/llm_internal/export/*.gguf" >/dev/null 2>&1; then
    echo "Done: export artifacts are in $OUT_DIR/llm_internal/export/"
    echo "Copy them locally and run:"
    echo "  ollama create homemade-llm -f $OUT_DIR/llm_internal/export/Modelfile"
    echo "  ollama run homemade-llm"
elif compgen -G "$OUT_DIR/llm_internal/checkpoints/checkpoint-*" >/dev/null 2>&1; then
    echo "Training didn't finish this session. Re-run this script (e.g. next"
    echo "week once the GPU quota resets) -- it will resume automatically"
    echo "from $OUT_DIR/llm_internal/checkpoints/."
else
    echo "No checkpoints or export found in output -- inspect $OUT_DIR/ and" >&2
    echo "the kernel logs (kaggle kernels output $KERNEL_ID -p $OUT_DIR)." >&2
    exit 1
fi
