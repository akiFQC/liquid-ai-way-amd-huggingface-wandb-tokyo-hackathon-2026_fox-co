# Audio fine-tuning track

> Haven't set up the kit yet? Start at the [main README](../../README.md#-tldr-your-first-run-in-5-commands) and come back here once `make validate-audio` is green.

As a starting point you can fine-tune **LFM2.5-Audio-1.5B** (speech-in, speech-out) on the Jenny TTS dataset. See the upstream [liquid-audio](https://github.com/Liquid4All/liquid-audio) reference recipe; the
`preprocess_jenny_tts.py` + `train.py` in its [`examples/`](https://github.com/Liquid4All/liquid-audio/tree/main/examples)
are the scripts this kit's flow mirrors.

## Workflow

```
your iterator → preprocess to disk → LFM2DataLoader → Trainer → checkpoint
                 (one-off, ~5-10 min        (~50 min on 1× A100 80GB
                 on a small GPU)             at the reference recipe)
```

## 1. Install audio dependencies

```bash
uv sync --extra audio
```

Brings in `liquid-audio` plus its dependencies (torchaudio, librosa, soundfile, etc.).

## 2. Fine-tune

The HF Jobs launcher runs fine-tuning via the HF compute platform so you can leverage stronger hardware:

```bash
HF_FLAVOR=a100-large ./scripts/audio/launch_hf_job.sh
```

If you have another training environment with a CUDA GPU you'd like to run the script in, the underlying Python script runs locally
(it's a self-contained [UV script](https://docs.astral.sh/uv/guides/scripts/); deps resolve on first run):

```bash
MAX_STEPS=100 DATASET_SLICE=200 uv run scripts/audio/train.py
```

**Bring your own data by writing your own iterator.** The training contract is an iterable that yields
`list[ChatMessage]` per sample. For TTS: a system turn carrying the voice prompt, a user turn with
the text, an assistant turn with the audio bytes. `TrainingSamples` in
[`scripts/audio/train.py`](../../scripts/audio/train.py) is that iterable: ~15 lines that stream the
Jenny dataset and yield that shape. Your dataset will look different, so rewrite `__iter__` for wherever
your data lives (a HF dataset, a folder of WAVs + a CSV of transcripts, anything). Filtering, column
mapping, multi-config handling: all just code in your iterator.

This kit's fine-tune recipe is **TTS-only**. Be sure to update your code accordingly if you're doing ASR or STS tasks.

### Reference recipe (production launcher defaults)

All overridable via env vars on the launcher (`./scripts/audio/launch_hf_job.sh`); the full list lives at the top of
[`scripts/audio/train.py`](../../scripts/audio/train.py).

| Env var | Default | Notes |
| --- | --- | --- |
| `MODEL_ID` | `LiquidAI/LFM2.5-Audio-1.5B` | Base checkpoint |
| `DATASET_SLICE` | `0` | `0` = full dataset; positive N caps to first N rows |
| `BATCH_SIZE` | `64` | Per-device batch |
| `MAX_STEPS` | `5000` | ~50 min on 1× A100 80GB |
| `LR` | `1e-4` | AdamW, linear warmup then no decay |
| `PUSH_TO_HUB` | _(unset)_ | If set, the trained checkpoint is pushed to that HF repo id |

There is no `DATASET` or `SYSTEM_PROMPT` env var: the data, and the voice prompt tied to
it, are both a code change (your iterator), not config. `TrainingSamples.SYSTEM_PROMPT` in the script holds
the example's Irish-voice prompt; rewrite it alongside `__iter__` for your own voice. Recipe constants that
rarely move (warmup=250, context length=256) live at the top of the script, edit them there.

**Preprocessing:** every HF Jobs run preprocesses shards from scratch into `$OUTPUT_DIR/preprocessed/` (no
cross-run cache; cap iteration time with `DATASET_SLICE`).

## 3. W&B logging

The training script's `WandbTrainer` (in [`scripts/audio/train.py`](../../scripts/audio/train.py))
subclasses the upstream `liquid_audio.trainer.Trainer` and overrides its `log()` to forward `train/loss` +
`train/lr` to W&B. With `WANDB_API_KEY` + `WANDB_PROJECT` / `WANDB_ENTITY` set[^1], the run shows up at
`https://wandb.ai/<entity>/<project>` automatically.

[^1]: The launcher forwards them as Job secrets

To hear the fine-tune, run [`scripts/run_eval_audio.py`](../../scripts/run_eval_audio.py) after training.
It synthesizes the same prompts with base vs fine-tuned weights and logs the pairs as `wandb.Audio` panels.

## 4. Recommended hardware sizing on HF Jobs

| Flavor | Suitable for | Notes |
| --- | --- | --- |
| `a100-large` (1× A100 80GB, $2.50/h) | Default recipe (bs=64, ctx=256) | $150 buys ~60 h, enough for 6+ full 5000-step runs |
| `l40sx1` (1× L40S 48GB, $1.80/h) | Lower batch size (bs=32, ctx=256) | Cheaper but check OOM on first 100 steps |
| `h200` (1× H200 141GB, $5.00/h) | Larger ablations (bs=128, ctx=512) | $150 buys ~30 h |


## 5. Load and deploy the fine-tuned model

`liquid_audio.trainer.Trainer` writes HF-compatible checkpoints to `--output-dir`. Loading and serving is the same as
the base model, just with your local path. The three patterns below mirror the upstream
[liquid-audio README](https://github.com/Liquid4All/liquid-audio#usage).

### 5a. Local TTS inference (Irish voice from the Jenny fine-tune)

The example `TrainingSamples` iterator trains the model to respond to **`"Perform TTS. Use the Irish female voice."`** as the
system prompt. Use that same prompt at inference to invoke the voice the model just learned (vs the four built-in
US/UK voices the base model already knows).

```python
from pathlib import Path

import soundfile as sf
import torch
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

# Point at your local checkpoint dir (or push to the Hub first, see §5b).
# Local dirs MUST be Path objects; liquid_audio treats a plain str as a
# Hub repo id and tries to download it.
CHECKPOINT = Path("/path/to/your/output-dir/final")

processor = LFM2AudioProcessor.from_pretrained(CHECKPOINT).eval()
model = LFM2AudioModel.from_pretrained(CHECKPOINT).eval().to("cuda")

chat = ChatState(processor)
chat.new_turn("system")
chat.add_text("Perform TTS. Use the Irish female voice.")
chat.end_turn()

chat.new_turn("user")
chat.add_text("Welcome to Hack the Liquid WAY. Audio fine-tuning is finally here.")
chat.end_turn()

chat.new_turn("assistant")

audio_out: list[torch.Tensor] = []
for t in model.generate_sequential(**chat, max_new_tokens=512, audio_temperature=0.8, audio_top_k=64):
    if t.numel() > 1:
        audio_out.append(t)

# Detokenize (drop the trailing end-of-audio frame). Mimi emits 24 kHz.
audio_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
waveform = processor.decode(audio_codes)
sf.write("tts_irish.wav", waveform.cpu().squeeze().numpy(), 24_000)
```

Note that the ASR (`"Perform ASR."`) and interleaved speech-to-speech (`"Respond with interleaved text and audio."`) are
capabilities trained in the base model. These capabilities may no longer work properly after fine-tuning, so we
recommend only using your fine-tuned model with the system prompt(s) it was trained on.

### 5b. Push the fine-tune to HuggingFace Hub

So your teammates (and the judges) can load the model with a one-line `from_pretrained("you/your-finetune")`.

The easy path: set `PUSH_TO_HUB=your-username/your-finetune` as a launcher arg and the training job pushes the
finished checkpoint itself (and stamps the commit SHA into the W&B run so `make verify` can prove it landed).

To push an existing local checkpoint manually, upload the `final/` directory. The training script makes it
self-contained (trained weights + config + tokenizer + Mimi codec + detokenizer):

```bash
hf auth login   # once, on the machine that holds the checkpoint
hf upload your-username/lfm25-audio-jenny-finetune /path/to/your/output-dir/final
```

Pushing makes the model loadable from any machine (the judging PC, the AMD Ryzen AI PC, a teammate's laptop) without
copying the checkpoint dir around. Set the repo to private during development; flip to public for the demo when you
want judges to inspect the model card.

### 5c. On-device demo (AMD Ryzen AI PC)

For the live demo at judging, the assigned AMD Ryzen AI PC runs the same `liquid-audio` package via the on-device
runbook in [`examples/on_device/`](../on_device/). On-device latency across the AMD SKUs varies, so pre-record a video
demo to complement your live demo.

## Japanese voice datasets

The default recipe trains on Jenny TTS (Irish female English). Try implementing a Japanese version with an open-license
JP speech corpus (e.g., [`reazon-research/reazonspeech`](https://huggingface.co/datasets/reazon-research/reazonspeech),
`load_dataset(..., "small")`) and rewrite `TrainingSamples.__iter__` in
[`scripts/audio/train.py`](../../scripts/audio/train.py) for its columns and voice prompt (config names, splits, and
per-row filtering are all ordinary Python in the iterator). Then submit with
`./scripts/audio/launch_hf_job.sh`.

> [!WARNING]
> Check the license for any dataset you use in this hackathon to make sure it permits derivative model releases before
> pushing to a public HF repo.

## Caveats

- **LoRA is not built-in, but it's doable.** The upstream Trainer does a full bf16 fine-tune; there's no
  automatic LoRA flag like HF's `Trainer`. To train a LoRA, inject adapters with `peft` (see the
  [PEFT docs](https://huggingface.co/docs/peft)) or wire it in by hand (~20 lines). Note that this is deliberately off
  the happy path so only do this if it makes sense for your project.
- **CUDA driver compatibility.** The default `torch+cu130` wheel crashes on HF Jobs A100s: their driver
  supported only up to CUDA 12.9 (`found version 12090`) as of 2026-06-05, and HF doesn't document the host
  version (so it can change). Pin `torch` to the `pytorch-cu126` index on Linux:
  ```toml
  # /// script
  # [[tool.uv.index]]
  # name = "pytorch-cu126"
  # url = "https://download.pytorch.org/whl/cu126"
  # explicit = true
  # [tool.uv.sources]
  # torch = [{ index = "pytorch-cu126", marker = "platform_system == 'Linux'" }]
  # ///
  ```
  Mac dev machines need the unpinned default; the `platform_system` marker handles both.
- **Dataset license.** Check the dataset license permits derivative model release before pushing to a public HF repo.
