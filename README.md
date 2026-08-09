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
3. **Eval** — held-out gate on tool-call structural accuracy + plain-chat pass rate (`llm_internal.eval.run_eval`), plus an independent tool-calling benchmark with fine-grained metrics and quality gates (`llm_internal.eval.run_benchmark`); both dispatch on their config's `backend` -- see "Evaluation" below
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
uv run ruff check .      # lint
uv run pyright           # type check (src/llm_internal/train and tests/train
                          # are excluded -- see pyproject.toml's [tool.pyright]
                          # comment for why)
uv run pytest            # unit tests + coverage report, no GPU required
```

CI (`.github/workflows/ci.yml`) runs exactly these three commands against
the locked environment (`uv sync --locked --extra dev`) on every push/PR.
It does not run model training or GPU/Metal-backed evaluation -- those need
real hardware; run them via the `scripts/run_on_*.sh` entry points below.

## Config

- `configs/data.yaml` — dataset source, revision pin, split ratios
- `configs/train.yaml` — base model + pinned revision, LoRA hyperparameters, training schedule, `backend`
- `configs/eval.yaml` — Hermes held-out split gate thresholds, `backend`
- `configs/benchmark_eval.yaml` — independent benchmark config (fine-tuned model), gate overrides, `backend`
- `configs/benchmark_eval_base.yaml` — same benchmark, pointed at the original base model, for comparison
- `configs/export.yaml` — export model/output dirs, quant level, `backend`

### Reproducibility

- **Seeds**: `configs/data.yaml`'s `seed` (split shuffling) and
  `configs/train.yaml`'s `seed` (LoRA init/training) are fixed for
  deterministic reruns; `configs/benchmark_eval*.yaml`'s `seed` is
  reserved for future sampling-based generation (current eval is greedy
  decoding, which is already deterministic).
- **Dataset revision**: `configs/data.yaml`'s `dataset_revision` pins
  `NousResearch/hermes-function-calling-v1` to an immutable commit sha —
  left unchanged by this work per the task constraints.
- **Base model revision**: `configs/train.yaml`'s `base_model_revision`
  pins `Qwen/Qwen3-1.7B` to an immutable commit sha (rather than the
  mutable `main` ref), threaded through every load site (Unsloth's
  `FastLanguageModel.from_pretrained`, `mlx_lm.convert`,
  `eval/generation.py`'s CUDA/MLX loaders). To pick up a newer release:
  check https://huggingface.co/Qwen/Qwen3-1.7B/commits/main for the new
  commit sha, update `base_model_revision` in both `configs/train.yaml`
  and `configs/benchmark_eval_base.yaml`, and re-run training + the
  base-vs-fine-tuned comparison.

## Evaluation

Two complementary evaluations exist:

1. **Held-out Hermes split** (`llm_internal.eval.run_eval`, `configs/eval.yaml`)
   — a deterministic slice of the training dataset itself
   (`data/processed/eval.jsonl`, produced by `data.prepare`'s stratified
   split). Cheap, fast, but *not independent*: it's drawn from the same
   distribution the model was fine-tuned on, so a high score here mostly
   confirms the fine-tune converged, not that it generalizes.
2. **Independent benchmark** (`llm_internal.eval.run_benchmark`,
   `configs/benchmark_eval.yaml`) — hand-authored prompts the model has
   never seen during fine-tuning, scored with fine-grained metrics and
   explicit quality gates. This is the primary reliability signal; the
   Hermes split remains as a cheap sanity check.

### Benchmark structure

`data/benchmark/cases.jsonl` — one JSON object per line, each a
self-contained test case:

```json
{
  "id": "single_tool_001",
  "category": "single_tool_selection",
  "description": "Only one relevant tool is available; model should call it with correct args.",
  "tools": [ {"type": "function", "function": {"name": "get_weather", "description": "...", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}} ],
  "messages": [ {"role": "system", "content": "...<tools>[...]</tools>"}, {"role": "user", "content": "What's the weather in Paris?"} ],
  "expects_tool_call": true,
  "expected_tool_calls": [ {"name": "get_weather", "arguments": {"city": "Paris"}} ]
}
```

`tools` uses the same OpenAI/JSON-Schema-style function-spec shape as the
Hermes training data, so cases render through the production chat
template unchanged. See `src/llm_internal/eval/benchmark.py`'s module
docstring for the full field reference (including `acceptable_tool_names`,
used for inherently ambiguous cases where more than one tool selection is
correct).

The initial dataset covers 26 cases across 10 scenario categories:
`single_tool_selection`, `no_tool_plain_chat`, `ambiguous_tool_request`,
`nonexistent_tool_request`, `multi_tool_choice`, `complex_arguments`,
`optional_nullable_arguments`, `nested_arguments`, `parallel_tool_calls`,
`must_not_call_tool`. Failure modes like hallucinated tool names/arguments,
missing required arguments, and incorrect argument values aren't separate
*case* categories (a prompt can't itself "hallucinate" -- only a model's
reply can) -- they're detected generically from any case's prediction by
the metrics below (`hallucinated_tool_name_rate`,
`hallucinated_argument_rate`, `missing_required_argument_rate`,
`argument_value_accuracy`).

**To add a case:** append one JSON line to `data/benchmark/cases.jsonl`
(or a new file, then list it under `benchmark_files` in
`configs/benchmark_eval.yaml`). No schema migration or code change
required; `llm_internal.eval.benchmark.load_benchmark` validates every
case on load (unique `id`, non-empty `messages`, `expected_tool_calls`
and/or `acceptable_tool_names` present whenever `expects_tool_call` is
`true`, etc.) and fails loudly on a malformed entry.

### Metrics

`llm_internal.eval.metrics` computes, globally and per-category:

| Metric | Question it answers |
|---|---|
| `tool_selection_accuracy` | Right tool(s) called, or rightly none? |
| `tool_call_precision` / `tool_call_recall` | Per-call precision/recall of predicted vs. expected tool names (parallel calls matched independently) |
| `false_positive_tool_rate` | Of cases expecting *no* call, how often did the model call one anyway? |
| `false_negative_tool_rate` | Of cases expecting a call, how often did the model call nothing? |
| `argument_name_accuracy` | Right argument *names* (F1 of predicted vs. expected keys, matched calls only) |
| `argument_value_accuracy` | Of correctly-named arguments, how many also have the right *value*? |
| `required_argument_accuracy` | Fraction of the tool schema's required arguments actually present |
| `schema_validity_rate` | Fraction of `<tool_call>` blocks that are valid JSON with `{"name": str, "arguments": dict}` |
| `exact_tool_call_match` | Strictest metric: prediction exactly equals expected (name + all argument values) |
| `plain_chat_pass_rate` | Deterministic reply-quality checks passed (see below) on cases expecting no call |
| `hallucinated_tool_name_rate` | Predicted calls naming a tool absent from the case's declared `tools` |
| `hallucinated_argument_rate` | Predicted argument names absent from the tool's schema `properties` |
| `missing_required_argument_rate` | Complement of `required_argument_accuracy` |

These are deliberately kept separate (never averaged into one score) so a
model can't hide an argument-value regression behind a good tool-selection
score, or vice versa -- matching the three questions "right tool?",
"structurally valid arguments?", "semantically correct values?".

### Plain-chat validation

`llm_internal.eval.plain_chat.check_plain_chat_response` replaces a bare
minimum-length check with deterministic, reproducible heuristics (no
LLM-as-judge): empty/too-short replies, an unexpected `<tool_call>` block,
leaked special tokens (`<|im_end|>` etc.) or protocol tags (`<tools>`,
`<tool_response>`), a leaked role prefix (`assistant:`), and degenerate
repetition (a decoding-loop failure mode: one word dominating a long
reply). Every applicable reason is reported, not just the first.

### Quality gates

`llm_internal.eval.gates.DEFAULT_GATES` defines a threshold and direction
per metric, split into **mandatory** (must pass; failure flips the CLI's
exit code) and **advisory** (reported, never blocking). See
`eval/gates.py`'s module docstring for the full rationale behind each
default; briefly: mandatory gates cover the properties a broken
integration can't tolerate (wrong tool selection, invalid JSON, missing
required arguments, plain-chat regressions, over/under-triggering,
hallucinated tool names), while `exact_tool_call_match`,
`argument_name_accuracy`, `argument_value_accuracy`, and
`hallucinated_argument_rate` stay advisory because free-text/date argument
values can be correctly phrased in more than one way at this benchmark's
size.

Override any threshold per-run via `configs/benchmark_eval.yaml`'s
`gate_overrides` (a flat `{metric_name: threshold}` map layered onto the
defaults; direction and mandatory-ness stay fixed).

### Evaluating a model

```bash
# Hermes held-out split (fast sanity check)
uv run python -m llm_internal.eval.run_eval

# Independent benchmark (primary signal) -- edit configs/benchmark_eval.yaml's
# model_dir first (defaults to checkpoints/, the local fine-tune output)
uv run python -m llm_internal.eval.run_benchmark
```

Both print a summary and exit non-zero if a mandatory gate fails (or the
Hermes-split thresholds aren't met), so either is CI-safe as a release
gate. `run_benchmark` additionally writes `benchmark_report.json` with the
full overall/per-category/gate breakdown.

### Comparing base vs. fine-tuned

```bash
uv run python -m llm_internal.eval.compare_models \
    configs/benchmark_eval_base.yaml configs/benchmark_eval.yaml
```

Runs the *exact same* benchmark cases through both models (base model
config first, fine-tuned second) and prints a table of base value,
fine-tuned value, absolute delta, and verdict (`REGRESSION` / `improved` /
`-`) per metric, plus which mandatory gates each model passes. Also writes
`comparison_report.json`. Exits non-zero if the fine-tuned model fails a
mandatory gate or regresses on any metric -- this is the check that
answers "did fine-tuning actually help tool-calling, and did it break
plain chat?"

### Interpreting a report

- **Overall metrics regressed but plain_chat_pass_rate held** — fine-tuning
  traded some general capability for tool-calling reliability as intended;
  check whether it's within the advisory-metric tolerance.
- **tool_selection_accuracy high but argument_value_accuracy low** — the
  model picks the right tool but gets values wrong (dates, names, units);
  inspect `benchmark_report.json`'s `case_results`-equivalent categories
  (`complex_arguments`, `nested_arguments`) for concrete failures.
- **false_positive_tool_rate above threshold** — the model over-triggers
  on `must_not_call_tool`/`no_tool_plain_chat` cases; a training-data or
  system-prompt issue, not an argument-formatting one.
- **schema_validity_rate below threshold** — the model is emitting
  malformed `<tool_call>` JSON; check `max_new_tokens` isn't truncating
  output, and inspect raw predictions for the failure pattern.

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
