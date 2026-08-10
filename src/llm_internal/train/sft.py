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
    with open(path, encoding="utf-8") as f:
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
    import torch

    # isort: off
    # Unsloth must be imported before trl: its patches rewrite trl's
    # internals at import time, and importing trl first leaves those
    # patches half-applied -- observed effect is Unsloth mutating
    # tokenizer.eos_token to a literal '<EOS_TOKEN>' placeholder
    # sentinel (see https://github.com/unslothai/unsloth/issues/2797).
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer
    # isort: on

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=True,
        revision=cfg.base_model_revision,
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

    # transformers' TrainingArguments now rejects the (previously
    # implicit) bf16 default on GPUs that lack real bf16 tensor cores
    # (Ampere+ only) -- Turing/T4 doesn't qualify, so pick explicitly
    # rather than let a version-dependent default choose for us.
    bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        save_steps=cfg.checkpoint_every_steps,
        eval_strategy="steps",
        eval_steps=cfg.checkpoint_every_steps,
        max_length=cfg.max_seq_length,
        dataset_text_field="text",
        seed=cfg.seed,
        bf16=bf16,
        fp16=not bf16,
        # Explicit even though it's what an unset (None) eos_token
        # falls back to anyway -- defensive against the import-order
        # sensitivity noted above.
        eos_token=tokenizer.eos_token,
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
