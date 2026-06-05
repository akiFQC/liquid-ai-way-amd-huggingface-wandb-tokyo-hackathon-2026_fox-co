# Text fine-tuning track

> Haven't set up the kit yet? Start at the [main README](../../README.md#-tldr-your-first-run-in-5-commands) and come back here once `make validate` is green.

Fine-tune **LFM2 / Liquid Nanos** on text-only tasks (translation, RAG, structured extraction, function calling,
chat) with TRL `SFTTrainer` + PEFT LoRA. Optional `PUSH_TO_HUB` writes the merged checkpoint back to your
namespace; [`scripts/run_eval.py`](../../scripts/run_eval.py) then compares base vs fine-tune side by side.

## Workflow

```
HF dataset → SFTTrainer + LoRA → merged ckpt → (optional) HF Hub → run_eval.py base-vs-finetune
              (LFM2-350M / 700M / 1.2B)          (PUSH_TO_HUB=user/repo)   (wandb.Table + Weave traces)
```

## 1. Install dependencies

The text-track deps are in the base `pyproject.toml` (no extra needed):

```bash
uv sync
```

Bundles `transformers`, `datasets`, `peft`, `trl`, `wandb`, `weave`, and a few helpers. Total venv is
~4 GB after install.

## 2. Pick a base model + dataset

**Base model**, via `MODEL_ID` env on the launcher (defaults to `LiquidAI/LFM2-350M`):

```bash
MODEL_ID=LiquidAI/LFM2-350M    ./scripts/text/launch_hf_job.sh   # small + fast for iteration
MODEL_ID=LiquidAI/LFM2-700M    ./scripts/text/launch_hf_job.sh   # next size up
MODEL_ID=LiquidAI/LFM2-1.2B    ./scripts/text/launch_hf_job.sh   # largest text variant
# LFM2 family + LFM2-VL only (share the same attention module names). For
# other architectures the LoRA target_modules list in scripts/text/train.py
# needs adjusting; see the PEFT docs for the right modules per architecture.
```

**Dataset**, default is `mlabonne/FineTome-100k-dedup` (general-purpose SFT corpus). Swap via `DATASET`:

```bash
DATASET=your-org/your-dataset DATASET_SLICE=3000 ./scripts/text/launch_hf_job.sh
```

`DATASET_SLICE=0` means use the full dataset; any positive integer caps to that many samples.

### Japanese datasets

Verified-on-Hub candidates for Tokyo-themed work. `scripts/text/train.py:_to_messages()` handles the column
shapes marked ✅ out of the box; the others need a small adapter (extend `_to_messages` to match).

| Dataset | Size | Shape | Auto-mapped |
|---|---|---|---|
| [`kunishou/databricks-dolly-15k-ja`](https://huggingface.co/datasets/kunishou/databricks-dolly-15k-ja) | 15K | Dolly (`instruction`/`context`/`response`) | ✅ |
| [`llm-jp/databricks-dolly-15k-ja`](https://huggingface.co/datasets/llm-jp/databricks-dolly-15k-ja) | 15K | Dolly | ✅ |
| [`fujiki/japanese_alpaca_data`](https://huggingface.co/datasets/fujiki/japanese_alpaca_data) | ~52K | Alpaca (`instruction`/`input`/`output`) | ✅ |
| [`izumi-lab/llm-japanese-dataset`](https://huggingface.co/datasets/izumi-lab/llm-japanese-dataset) | ~9M | Alpaca | ✅ |
| [`shi3z/ja_conv_wikipedia_orion14B_100K`](https://huggingface.co/datasets/shi3z/ja_conv_wikipedia_orion14B_100K) | 100K | ShareGPT (`conversations`) | ✅ |
| [`Aratako/Synthetic-JP-EN-Coding-Dataset-Magpie-69k`](https://huggingface.co/datasets/Aratako/Synthetic-JP-EN-Coding-Dataset-Magpie-69k) | 69K | Native (`messages`), coding-heavy | ✅ |
| [`shisa-ai/shisa-v2-sharegpt`](https://huggingface.co/datasets/shisa-ai/shisa-v2-sharegpt) | ~270K | ShareGPT | ✅ |
| [`CohereForAI/aya_dataset`](https://huggingface.co/datasets/CohereForAI/aya_dataset) (filter `language="Japanese"`) | ~200K total | Aya (`inputs`/`targets`) | ✅ |
| [`kunishou/oasst1-89k-ja`](https://huggingface.co/datasets/kunishou/oasst1-89k-ja) | 89K | OASST tree (`message_id`/`parent_id`/`role`/`text_ja`) | ❌ tree flatten needed |
| [`LiquidAI/OHF-Voice-audio-20260504`](https://huggingface.co/datasets/LiquidAI/OHF-Voice-audio-20260504) | 55K | Function-calling (`audio_chat` + `text_chat`) | ❌ custom adapter needed |

Switch via env on the launcher:

```bash
DATASET=kunishou/databricks-dolly-15k-ja DATASET_SLICE=2000 ./scripts/text/launch_hf_job.sh
```

### Datasets with non-standard column names

If your dataset doesn't match any of the 5 auto-recognized shapes (`messages`, `conversations`, `instruction+output`, `instruction+response`, `inputs+targets`), use `DATASET_MAPPER` to point at the right columns explicitly:

```bash
# {question, answer} columns
DATASET=your-org/qa-data \
DATASET_MAPPER="user=question,assistant=answer" \
  ./scripts/text/launch_hf_job.sh

# {prompt, completion, system_prompt} columns (system role optional)
DATASET=your-org/prompted-data \
DATASET_MAPPER="user=prompt,assistant=completion,system=system_prompt" \
  ./scripts/text/launch_hf_job.sh
```

The mapper is checked before the auto-shapes, so it also overrides an auto-shape when you want different columns. Both **format errors** (`bogus=col`, missing `user` or `assistant`) and **column-name typos** (`assistant=mispelled_col`) fail before `wandb.init` and model download: the launcher validates the first 1024 rows through `_to_messages` first, so a bad mapper costs ~1-2 s of CPU instead of ~90 s of remote startup plus a half-initialized W&B run. A bad row *past* that prefix still fails, just after `wandb.init` (a deterministic `ds.map` error before `trainer.train`).

Check each dataset's license before pushing a derivative fine-tune to a public HF repo.

## 3. Fine-tune

The production path is HF Jobs. The launcher reads `.env`, sets a sane `--timeout`, and forwards the credentials as container secrets:

```bash
./scripts/text/launch_hf_job.sh                                        # defaults
HF_FLAVOR=a100-large MAX_STEPS=500 ./scripts/text/launch_hf_job.sh     # bigger run
```

`examples/text/text_finetune_walkthrough.ipynb` is the pedagogical narrative notebook (open it in Jupyter / VS Code / Cursor / PyCharm); it shares the recipe but isn't what HF Jobs runs.

### Reference recipe (production launcher defaults)

All overridable via env vars on the launcher; no source edits needed.

| Env var | Default | Notes |
| --- | --- | --- |
| `MODEL_ID` | `LiquidAI/LFM2-350M` | LFM2 family + LFM2-VL only; full-precision (not Unsloth bnb-4bit). Other architectures need a custom LoRA `target_modules` in `scripts/text/train.py`. |
| `DATASET` | `mlabonne/FineTome-100k-dedup` | ShareGPT-format SFT corpus |
| `DATASET_SLICE` | `1024` | First N rows; `0` = full dataset |
| `MAX_STEPS` | `200` | Training step count; ~10 min on an A100 |
| `BATCH_SIZE` | `4` | Per-device batch (× grad accum 4, effective bs=16) |
| `LR` | `2e-4` | LoRA learning rate (higher than full-FT because layers are small) |
| `PUSH_TO_HUB` | _(unset)_ | If set, the merged checkpoint is pushed to that HF repo id |

The LoRA recipe itself (r=16, alpha=32, dropout=0, grad accum 4) is locked in the script per the
[Liquid AI Unsloth recipe](https://docs.liquid.ai/lfm/fine-tuning/unsloth); edit
[`scripts/text/train.py`](../../scripts/text/train.py) if you have a research reason to deviate.

The full list is documented at the top of [`scripts/text/train.py`](../../scripts/text/train.py).

## 4. W&B + Weave logging

Training ([`scripts/text/train.py`](../../scripts/text/train.py)) logs scalars via TRL's
`report_to=["wandb"]` (loss, lr, gradient norms, throughput) plus system metrics (GPU util, memory)
collected automatically by W&B's hardware logger.

Evaluation ([`scripts/run_eval.py`](../../scripts/run_eval.py)) is where the rest of the observability
lives: with `--wandb` it logs a base-vs-finetune `comparison_table` and captures each generation as a
`@weave.op` trace at `wandb.ai/<entity>/<project>/weave`.

To add custom Weave-traced scorers (BLEU, ROUGE, an LLM-as-judge, etc.), wrap your scorer in `@weave.op` and call it
from your eval loop; patterns and examples at [weave-docs.wandb.ai](https://weave-docs.wandb.ai/).

## 5. Hardware sizing on HF Jobs

| Flavor | Suitable for | Notes |
| --- | --- | --- |
| `t4-medium` (1× T4 16GB, $0.60/h) | LFM2-350M Q4 fine-tune (walkthrough) | $150 buys ~250 h |
| `a10g-large` (1× A10G 24GB, $1.50/h) | LFM2-700M Q4 fine-tune | $150 buys ~100 h |
| `l40sx1` (1× L40S 48GB, $1.80/h) | LFM2-1.2B Q4 or LoRA | $150 buys ~83 h |
| `a100-large` (1× A100 80GB, $2.50/h) | LFM2-1.2B full bf16 fine-tune | $150 buys ~60 h |

The Q4 sizings above apply to the **walkthrough's** Unsloth + bnb-4bit path. The production launcher
(`scripts/text/train.py`) uses full bf16 + PEFT LoRA; at `r=16` the LoRA adapter is small enough that A100
80 GB easily handles LFM2-1.2B at bs=4. Use the walkthrough's Q4 path if you want LFM2-1.2B to fit on T4 / L40S.

## 6. Load and deploy the fine-tuned model

Both paths write HF-format checkpoints: the launcher saves the merged (LoRA-folded) model to `OUTPUT_DIR`; the
walkthrough's Unsloth path saves Unsloth-format files alongside the HF-compatible ones in `outputs/`. Three
deployment patterns:

### 6a. Local inference

The kit's working local inference path is [`scripts/run_eval.py`](../../scripts/run_eval.py) (base vs fine-tune
comparison) and [`examples/demo/text_chat.py`](../demo/text_chat.py) (Gradio chat UI). Both apply the canonical LFM2
sampling (`temperature=0.3`, `min_p=0.15`, `repetition_penalty=1.05`) and the chat template with
`return_dict=True` (transformers 5.x). Copy from either to a new script if you need something custom.

### 6b. Push to HuggingFace Hub

```bash
hf auth login                                                 # one-time on the training machine
hf upload your-username/lfm2-350m-jp-mt outputs/lfm2-350m-finetune
```

Or push from the notebook with `HfApi.upload_folder`. Assemble the checkpoint locally with `save_pretrained`, then upload the directory in one commit, so a mid-upload failure can't leave a half-published repo (weights but no tokenizer):

```python
from huggingface_hub import HfApi

# Stage the checkpoint locally.
out = "outputs/lfm2-350m-finetune"
model.save_pretrained(out)
tok.save_pretrained(out)

# Single-commit upload: atomic, no partial-failure risk.
api = HfApi()
api.create_repo("your-username/lfm2-350m-jp-mt", private=True, exist_ok=True)
api.upload_folder(folder_path=out, repo_id="your-username/lfm2-350m-jp-mt",
                  commit_message="fine-tune checkpoint")
```

Set the launcher's `PUSH_TO_HUB=your-username/your-finetune` env to push the merged checkpoint automatically when
training finishes (the HF Jobs path does this; no extra step needed).

### 6c. On-device demo (AMD Ryzen AI PC)

LFM2 has llama.cpp `lfm2` architecture support and FastFlowLM `.q4nx` NPU offload (for LFM2-1.2B on Strix-class
silicon). The full GGUF-convert + llama-cli runbook lives in
[`examples/on_device/`](../on_device/); see §2a for text models.

## Caveats

- **Pedagogy vs production.** The `.ipynb` walkthrough is for interactive learning; the production launcher
  (`scripts/text/train.py` via `./scripts/text/launch_hf_job.sh`) is what HF Jobs uploads. Edit the launcher for
  real fine-tunes.
- **`model.eval()` after `trainer.train()`.** TRL's `SFTTrainer` leaves the model in training mode; call
  `.eval()` before any `generate()` in your own loops (dropout left on quietly degrades sampling), or use
  `merge_and_unload()` to fold the adapter into the base weights.
- **`return_dict=True` for the chat template.** transformers 5.x's `apply_chat_template` returns a `BatchEncoding`,
  not a `Tensor`. Pass `return_dict=True` and call `model.generate(**inputs, ...)`.
- **LoRA `alpha = 2 × r`.** `lora_alpha=32` for `r=16` per the
  [official Liquid AI Unsloth recipe](https://docs.liquid.ai/lfm/fine-tuning/unsloth).
- **Smoke tests need real data + LoRA.** Tiny full-FT runs destroy LFM2's chat behavior. Use a real slice
  (e.g., `mlabonne/FineTome-100k-dedup` 256+ samples) and LoRA.
- **Dataset license.** Check each dataset's license permits derivative model release before pushing to a public HF repo.
