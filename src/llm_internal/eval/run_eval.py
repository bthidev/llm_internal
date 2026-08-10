"""Held-out eval gate. `evaluate_examples` is pure (takes pre-generated
predictions); `generate_predictions` (CUDA/transformers) and
`generate_predictions_mlx` (MLX) each require the trained model loaded on
real hardware, and delegate the actual load/generate work to
eval/generation.py (shared with eval/run_benchmark.py). `_load_and_generate`
dispatches between them based on EvalConfig.backend."""
from __future__ import annotations

import sys

from llm_internal.eval.config import EvalConfig, load_eval_config
from llm_internal.eval.generation import generate_cuda, generate_mlx, load_cuda_model, load_mlx_model, render_prompt
from llm_internal.eval.scoring import (
    Report,
    aggregate_results,
    score_plain_chat_example,
    score_tool_call_example,
    summarize_tool_call_failures,
)
from llm_internal.train.sft import load_split


def evaluate_examples(examples: list[dict], predictions: list[str], cfg: EvalConfig) -> Report:
    if len(examples) != len(predictions):
        raise ValueError(f"examples ({len(examples)}) and predictions ({len(predictions)}) length mismatch")

    per_example_results = []
    for example, predicted_text in zip(examples, predictions, strict=True):
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
    greedily generate the model's reply."""
    prompts = [render_prompt(example["messages"][:-1], tokenizer) for example in examples]
    return generate_cuda(prompts, model, tokenizer, cfg.max_new_tokens)


def generate_predictions_mlx(examples: list[dict], model_dir: str, cfg: EvalConfig) -> list[str]:
    """Metal-only: mirrors generate_predictions but loads/generates via
    mlx_lm."""
    model, tokenizer = load_mlx_model(model_dir)
    prompts = [render_prompt(example["messages"][:-1], tokenizer) for example in examples]
    return generate_mlx(prompts, model, tokenizer, cfg.max_new_tokens)


def _load_and_generate(examples: list[dict], cfg: EvalConfig) -> list[str]:
    """Dispatches to the CUDA (transformers) or MLX prediction path based
    on cfg.backend."""
    if cfg.backend == "mlx":
        return generate_predictions_mlx(examples, cfg.model_dir, cfg)

    model, tokenizer = load_cuda_model(cfg.model_dir)
    return generate_predictions(examples, model, tokenizer, cfg)


def main() -> None:
    cfg = load_eval_config("configs/eval.yaml")
    examples = load_split(cfg.eval_file)

    predictions = _load_and_generate(examples, cfg)
    report = evaluate_examples(examples, predictions, cfg)

    if report.tool_call_accuracy < cfg.tool_call_accuracy_threshold:
        print(summarize_tool_call_failures(examples, predictions))
    print(f"tool_call_accuracy={report.tool_call_accuracy:.3f} "
          f"plain_chat_pass_rate={report.plain_chat_pass_rate:.3f} passed={report.passed}")
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
