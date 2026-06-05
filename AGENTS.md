# AGENTS.md: AI-agent reference for extending this kit

You're an AI coding agent (Claude Code, Cursor, Copilot, ...) helping a hackathon participant extend the **Hack the Liquid WAY** starter kit. This is your quick-reference: read [`README.md`](README.md) first for the human walkthrough, then return here for "where do I edit X" / "how do I add Y" lookups.

If a recipe or hyperparameter is named here, it's **load-bearing**: quote it verbatim. Drift from the defaults is the #1 cause of doom-loop output and wasted credit.

## What this repo is

| Track | Model | Entry points |
|---|---|---|
| **Text** | `LiquidAI/LFM2-350M` (or 700M / 1.2B) | [`examples/text/text_finetune_walkthrough.ipynb`](examples/text/text_finetune_walkthrough.ipynb) (pedagogy), production launcher → [`scripts/text/train.py`](scripts/text/train.py) |
| **Audio** | `LiquidAI/LFM2.5-Audio-1.5B` | [`examples/audio/audio_finetune_walkthrough.ipynb`](examples/audio/audio_finetune_walkthrough.ipynb), production launcher → [`scripts/audio/train.py`](scripts/audio/train.py) (also runs locally on a CUDA box) |
| **Demo UIs** | n/a | [`examples/demo/text_chat.py`](examples/demo/text_chat.py) (Gradio) |
| **On-device** | n/a | [`examples/on_device/README.md`](examples/on_device/README.md) (llama.cpp + Vulkan / FastFlowLM NPU / liquid-audio) |
| **Eval** | n/a | [`scripts/run_eval.py`](scripts/run_eval.py) (text) + [`scripts/run_eval_audio.py`](scripts/run_eval_audio.py) (audio): base-vs-finetune side-by-side |
| **Submission** | n/a | [`DEMO_DAY.md`](DEMO_DAY.md) |

Teams submit training to **HuggingFace Jobs** ($150 credit per team, claimed with the link shared with registered teams). Production deployment targets are AMD Ryzen AI PCs (on-device) and the HF Hub.

## Canonical recipes: quote, don't invent

| Topic | Source | Verbatim |
|---|---|---|
| **LFM2 generation** | [LFM2-350M model card](https://huggingface.co/LiquidAI/LFM2-350M) | `do_sample=True, temperature=0.3, min_p=0.15, repetition_penalty=1.05` |
| **LFM2 chat template** | Same | `apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", tokenize=True, return_dict=True)`; in transformers 5.x always pass `return_dict=True` so `model.generate(**inputs, ...)` gets a `BatchEncoding`, not a raw tensor |
| **LFM2 LoRA (Unsloth)** | [docs.liquid.ai/lfm/fine-tuning/unsloth](https://docs.liquid.ai/lfm/fine-tuning/unsloth) | `r=16, lora_alpha=32, lora_dropout=0, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` |
| **LFM2 LoRA (TRL)** | [docs.liquid.ai/lfm/fine-tuning/trl](https://docs.liquid.ai/lfm/fine-tuning/trl) | Same r/alpha; `lora_dropout=0.05`; attention-only targets (`q/k/v/o`); `learning_rate=2e-4`, `per_device_train_batch_size=4`, `gradient_accumulation_steps=4`, `num_train_epochs=3`, `bf16=True` |
| **LFM2.5-Audio fine-tune** | [liquid-audio](https://github.com/Liquid4All/liquid-audio) | `bs=64, ctx=256, lr=1e-4, max_steps=5000, warmup_steps=250` (full bf16, no LoRA) |
| **LFM2.5-Audio TTS sampling** | [liquid-audio README](https://github.com/Liquid4All/liquid-audio#tts) | `audio_temperature=0.8, audio_top_k=64` (sequential generation); `audio_temperature=1.0, audio_top_k=4` for interleaved STS |

If the participant asks you to "try `top_k=50`" or "drop the repetition penalty", push back: those values are pinned to model-card defaults for a reason. Drift produces doom-loop output. If they have a defensible reason (research, ablation), document why before changing.

## How the participant runs the kit

```bash
make install                                   # one-time: uv sync
$EDITOR .env                                   # paste HF_TOKEN (WRITE scope) + WANDB_API_KEY/ENTITY/PROJECT
make validate                                  # auth + import probe (~10-20s)
make smoke-text                                # ~3 min local LFM2-350M smoke; no W&B / Hub
./scripts/text/launch_hf_job.sh                # text fine-tune on HF Jobs
./scripts/audio/launch_hf_job.sh               # audio fine-tune on HF Jobs
```

Each launcher prints its flavor + timeout (billing stops at the timeout; rates at [huggingface.co/docs/hub/jobs-pricing](https://huggingface.co/docs/hub/jobs-pricing)) and gates submission on `PUSH_TO_HUB` writeability.

## Customization surface: what to edit for which goal

| Goal | How |
|---|---|
| Change base model | `MODEL_ID=LiquidAI/LFM2-700M ./scripts/text/launch_hf_job.sh`. LFM2 family + LFM2-VL share the same attention module names so the hardcoded LoRA `target_modules` in [`scripts/text/train.py`](scripts/text/train.py) just works. Other architectures (Mistral, Phi, Qwen) need a custom `target_modules` list; see PEFT docs. |
| Change dataset (standard shape) | `DATASET=user/dataset ./scripts/text/launch_hf_job.sh`. Auto-handled if the row layout is `messages`, `conversations`, `instruction+output[+input]`, `instruction+response[+context]`, or `inputs+targets`. |
| Dataset with non-standard columns (text) | `DATASET_MAPPER="user=question,assistant=answer" ./scripts/text/launch_hf_job.sh` (optional `system=col_c`). Validates at startup before GPU time is consumed. |
| Train audio on your own dataset | Rewrite `TrainingSamples.__iter__` in [`scripts/audio/train.py`](scripts/audio/train.py) to yield `list[ChatMessage]` (system voice prompt, user text, assistant audio bytes) from wherever the data lives. No env knobs, no HF-format requirement. TTS-focused; ASR / STS fine-tuning is out of scope. |
| Hyperparameters | `MAX_STEPS=N BATCH_SIZE=N LR=N LORA_R=N ./scripts/text/launch_hf_job.sh`. Full env-var list at the top of [`scripts/text/train.py`](scripts/text/train.py) / [`scripts/audio/train.py`](scripts/audio/train.py), tiered into Common overrides + Advanced. |
| Push fine-tune to the Hub | `PUSH_TO_HUB=<your-hf-username>/repo ./scripts/text/launch_hf_job.sh`. **HF_TOKEN must have write access to the target repo** (default tokens are read-only). The launcher runs a writeability probe at submit time so credit isn't spent on a run whose post-train push will 403. For new (not-yet-existing) targets, opt into auto-create with `CREATE_PUSH_TARGET=1`. See [TROUBLESHOOTING.md → 403 Forbidden when pushing](TROUBLESHOOTING.md). |
| Truly unsupported dataset shape | Extend `_to_messages` in [`scripts/text/train.py`](scripts/text/train.py). |
| Custom training callbacks / loss / etc. | Edit `scripts/text/train.py` (text) or `scripts/audio/train.py` (audio) directly; they're self-contained UV scripts. Deps are declared in the `# /// script` header at the top of each file; add a Python dependency by editing that block, no `pyproject.toml` change needed. |
| Different sampling | **Don't.** Locked to the LFM2 model card. If you have a defensible reason (research, ablation), document why. |
| New Gradio demo | Lazy-load pattern: model loading goes inside `main()`; pure functions take the loaded objects as parameters. See [`examples/demo/text_chat.py`](examples/demo/text_chat.py) for the pattern. |
| Deploy to a HuggingFace Space | Step-by-step in [`examples/demo/README.md`](examples/demo/README.md): create a Gradio Space, copy the demo to `app.py`, drop a small `requirements.txt`, set `MODEL_ID` as a Space secret, push. |

## Make targets the participant runs

| Command | Purpose |
|---|---|
| `make install` | `uv sync` |
| `make validate` | Auth + import probe ([`scripts/shared/_validate_env.py`](scripts/shared/_validate_env.py)) |
| `make validate-audio` | Same as `make validate` plus a probe of the `[audio]` extra (liquid_audio) |
| `make smoke-text` | Load LFM2-350M and generate 16 tokens locally (no W&B, no Hub) |
| `make smoke-audio` | Synthesize ~1.3s TTS, CUDA box only (liquid_audio's decode is CUDA-only upstream; fails fast with guidance on Macs) |
| `make smoke-eval` | `run_eval.py --wandb` (base vs base on Mac MPS, ~3 min, real W&B logging) |

## Pitfalls: quick reference

Full table in [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md). The most likely to hit while extending:

- **Doom-loop output post-LoRA** → `trainer.model.eval()` BEFORE any `model.generate(...)`. `trainer.train()` leaves dropout active; sampling against a dropout-enabled model destroys the distribution.
- **`apply_chat_template` returns weird object** → `transformers` 5.x: pass `return_dict=True` and call `model.generate(**inputs, ...)` (note the `**`), not `model.generate(inputs, ...)`.
- **`PUSH_TO_HUB` → `403 Forbidden`** → re-issue HF_TOKEN at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with write access to the target repo, OR change `PUSH_TO_HUB` to a repo you can write to. The launcher gates this pre-submit, so you'll usually see the error locally before any credit is spent.
- **`NVIDIA driver too old (found version 12090)`** → the default torch wheel targets CUDA 13, but the HF Jobs A100 driver supported only up to CUDA 12.9 (`12090`) as of 2026-06-05 (HF doesn't document the host CUDA version, so it can change). New UV scripts need the `pytorch-cu126` index pattern from the existing launchers' inline metadata (`[[tool.uv.index]]` + `marker = "platform_system == 'Linux'"`).

## When you extend the kit

- **UV scripts (`scripts/{text,audio}/train.py`)** declare their own dependencies inline in the `# /// script` block at the top of the file. To add a Python dependency, add it to that block, not to `pyproject.toml`. The launcher uploads the script to HF Jobs which resolves those deps in the container.
- **Keep imported helpers import-safe**: [`scripts/text/train.py`](scripts/text/train.py) keeps `_to_messages` / `_parse_dataset_mapper` at module level with the heavy training imports inside `main()`, because the launcher's dataset preflight (`scripts/shared/_check_dataset.py`) imports those helpers without wanting torch. Mirror that split if you add helpers that other scripts import.
- **Match the canonical sampling + LoRA defaults** above unless you have a documented research reason to deviate. The recipes are pinned to model cards / upstream sources and tested in the kit.

## Verifying a real training run

`./scripts/text/launch_hf_job.sh` prints a `WANDB_RUN_NAME` (or you set one via env). After the job finishes:

```bash
python scripts/verify_run.py --run-name <name> \
    --require-metrics train/loss \
    --require-hf-repo <your-hf-username>/your-finetune
```

Returns exit 0 only if W&B scalars + tags + system metrics + HF repo all confirm. "Script exited 0" alone isn't enough; confirm metrics actually landed before claiming success. (Weave traces come from the eval scripts, not training; see [`scripts/run_eval.py`](scripts/run_eval.py).)
