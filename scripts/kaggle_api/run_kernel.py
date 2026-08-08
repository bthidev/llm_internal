"""Kaggle script-kernel entrypoint for the CUDA training pipeline.

Pushed to Kaggle by scripts/run_on_kaggle_api.sh via `kaggle kernels push`
-- not meant to be run manually inside the Kaggle notebook UI (use
scripts/run_on_kaggle.sh for that interactive workflow instead).

Clones this repo, seeds checkpoints/ from a mounted resume dataset if one
is attached (see kernel-metadata.json's dataset_sources, wired
automatically by run_on_kaggle_api.sh), then runs the same pipeline as
run_on_kaggle.sh. Everything left under /kaggle/working becomes this
kernel run's output, downloadable locally via `kaggle kernels output`.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

# Edit before the first push: must be reachable with internet access
# (public repo, or an https URL embedding a token for a private repo).
REPO_URL = "https://github.com/bthidev/llm_internal.git"

REPO_DIR = Path("/kaggle/working/llm_internal")
INPUT_DIR = Path("/kaggle/input")


def find_resume_checkpoints() -> str | None:
    """Look for a mounted dataset holding a prior run's checkpoints/."""
    if not INPUT_DIR.exists():
        return None
    for dataset_dir in sorted(INPUT_DIR.iterdir()):
        candidate = dataset_dir / "checkpoints"
        if candidate.is_dir() and any(candidate.glob("checkpoint-*")):
            return str(candidate)
    return None


def main() -> None:
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True
    )

    env = os.environ.copy()
    resume_from = find_resume_checkpoints()
    if resume_from:
        env["RESUME_FROM"] = resume_from
        print(f"Resuming from mounted dataset checkpoints: {resume_from}")
    else:
        print("No resume dataset mounted -- starting fresh.")

    subprocess.run(
        ["bash", "scripts/run_on_kaggle.sh"], check=True, cwd=str(REPO_DIR), env=env
    )


if __name__ == "__main__":
    main()
