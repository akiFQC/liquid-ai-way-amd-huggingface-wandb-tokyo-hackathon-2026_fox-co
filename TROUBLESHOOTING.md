# Troubleshooting

Common error messages from the kit, mapped to root cause and fix. If you hit something not listed here, check `make validate` first; most setup-side breakage shows up there.

## Setup

### `uv: command not found`

uv isn't installed yet. See [README → Prerequisites](README.md#-prerequisites) for the install instructions, or grab it from the [Astral installer page](https://docs.astral.sh/uv/getting-started/installation/).

### `make: command not found` (Windows)

GNU make isn't on PATH. Either install it via your package manager (e.g. `winget install ezwinports.make`) or run the underlying commands directly: `uv sync` instead of `make install`, etc. See the `Makefile` for the full command list.

### `make install` succeeds but `make validate` says `import transformers failed`

Stale `.venv` from a previous Python upgrade. The venv's `python3` symlinks at the old interpreter, which uv can't run. Fix:

```bash
rm -rf .venv
make install
```

### `make validate` says `HF_TOKEN unset` even though it's in `.env`

Your `.env` uses `KEY = "value"` (spaces around `=`) and your shell can't `source` it directly. Run via the project's tolerant loader:

```bash
bash -c 'source scripts/shared/_load_env.sh && _load_env .env && env | grep HF_TOKEN'
```

The Makefile targets (`make validate`, `make smoke-eval`, etc.) already use the loader; only ad-hoc shell sourcing tripped on the format.

### Launcher refuses to submit: "HF_TOKEN cannot write &lt;target&gt;"

Two possible root causes, both surfaced in the error message:

1. **Token is read-only**: Default fine-grained tokens don't include `repos:write`. Re-issue at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with **write access to the target namespace**.
2. **Token has write scope, but to a different namespace**: A token issued with write access to `your-username` but `PUSH_TO_HUB=other-org/repo` will 403. Either re-issue the token with write access to `other-org`, or change `PUSH_TO_HUB` to your own namespace or an org you belong to.

The launchers (`scripts/text/launch_hf_job.sh`, `scripts/audio/launch_hf_job.sh`) probe `PUSH_TO_HUB` writeability before submission, so read-only-token and wrong-namespace failures are caught at submit time instead of 1h later after training completes. Two modes depending on whether the target already exists:

**Existing target → repo-scoped tag probe.** The probe creates a disposable `hackathon-write-probe-<8hex>` tag on your `PUSH_TO_HUB` repo via the HF Hub tag API, then deletes it. Catches HF org / resource-group setups where you have access to specific repos but not namespace-wide create rights, plus any repo-level write restriction (locked / archived / collaborator-only) that a namespace-only probe would miss. If the DELETE step fails, the probe fails closed with the leaked tag's exact URL so you can remove it manually from the HF Hub UI before re-running.

**Missing target → namespace-sibling probe.** If `PUSH_TO_HUB` points at a repo that doesn't exist yet, the probe creates + deletes a disposable `<namespace>/hackathon-probe-<8hex>` sibling repo to confirm you have namespace-create rights; the actual `PUSH_TO_HUB` repo is auto-created at end-of-training via `create_repo(exist_ok=True)`. The launcher's banner echoes the resolved target before submit so typos (`my-finetnue` vs `my-finetune`) are visible there.

## Training submission

### `python file.py` → `SyntaxError` near `%pip` or `%cd`

You exported a Jupyter notebook to `.py` and ran it as a regular script. `%pip`, `%cd`, `%%capture` are Jupyter magics (valid only inside a notebook kernel). Either run the `.ipynb` in Jupyter / VS Code, or strip the `%`-prefixed lines manually before `python yourscript.py`.

### `The NVIDIA driver on your system is too old (found version 12090)`

You're pulling a default `torch` wheel built for CUDA 13, but the HF Jobs A100 driver supported only up to CUDA 12.9 (`found version 12090`) as of 2026-06-05 (HF doesn't document the host CUDA version, so it can change). Pin torch to the cu126 index in your UV script's inline metadata:

```python
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu126", marker = "platform_system == 'Linux'" }]
```

Both production launchers already do this; only custom UV scripts need to add it.

### `hf jobs run: argument --secret: invalid choice`

You typed `--secret`; the flag is `--secrets` (plural). `--secrets KEY=VALUE` to pass an explicit value, `--secrets KEY` to pull from your local environment.

## Inference / fine-tune output

### LFM2 emits `"TokTokTokTok..."` (or similar repetition)

Greedy decode + missing repetition penalty + no chat template. Use the canonical recipe from the [LFM2-350M model card](https://huggingface.co/LiquidAI/LFM2-350M):

```python
inputs = tok.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=True,
    return_dict=True,
).to(model.device)
gen = model.generate(
    **inputs,
    do_sample=True,
    temperature=0.3,
    min_p=0.15,
    repetition_penalty=1.05,
    max_new_tokens=512,
)
```

This is hardcoded in `examples/demo/text_chat.py` and `scripts/run_eval.py`; copy from there if you're writing a new inference script.

### `AttributeError` in `model.generate`: `inputs_tensor.shape` fails

`transformers` 5.x changed `apply_chat_template` to return `BatchEncoding` (dict-like) instead of a raw `Tensor`. Pass `return_dict=True` and use `model.generate(**inputs, ...)` (note the `**`), not `model.generate(inputs, ...)`.

### Post-LoRA inference doom-loops even with canonical sampling

`trainer.train()` leaves the model in **training mode**; dropout is active during `generate()`, which destroys the sampling distribution. Toggle eval mode explicitly:

```python
trainer.model.eval()                # <-- mandatory before any generation
out = trainer.model.generate(...)
```

Unsloth's `FastModel.for_inference(model)` does this automatically; raw PEFT + `SFTTrainer` (our path) does not. This bug is the single most common cause of "fine-tuning didn't help" reports.

### Want a portable checkpoint without the LoRA adapter

Merge the adapter into the base weights, then save:

```python
merged = trainer.model.merge_and_unload()
merged.eval()
merged.save_pretrained("./out")
tok.save_pretrained("./out")
```

The production launcher already does this; your `PUSH_TO_HUB` target gets the merged weights, not a LoRA-adapter-on-top.

## W&B / Weave

### Audio training runs but no W&B scalars appear

`liquid_audio.trainer.Trainer.log()` only prints to stdout; it doesn't call `wandb.log()`. The audio launcher subclasses it with a `WandbTrainer` that adds the bridge. If you're running the upstream `liquid-audio/examples/train.py` directly, you'll need to add that subclass yourself. See `scripts/audio/train.py:WandbTrainer` for the pattern.

### W&B run shows tags but zero metric samples

`wandb.init()` succeeded but `wandb.log()` was never called (or the trainer's `report_to=` arg doesn't include `"wandb"`). Confirm via:

```bash
python scripts/verify_run.py --run-name <your-run> --require-metrics train/loss
```

If `verify_run` says `metric 'train/loss' has zero non-null samples`, the trainer didn't push scalars. Check `trl.SFTConfig(report_to=["wandb"])` is set, or your custom trainer's `log()` actually calls `wandb.log()`.

### Weave URL works but trace shows zero calls

Either `weave.init()` wasn't called, or your inference function isn't decorated with `@weave.op`. In this kit, Weave traces come from the eval scripts (training only logs scalars); see `scripts/run_eval.py` for the pattern.

## Dataset shape

### `ValueError: Dataset row has no recognized shape. Got columns: [...]`

`scripts/text/train.py:_to_messages` covers 5 standard chat-format shapes (messages, conversations, instruction+output, instruction+response, inputs+targets). Your dataset's columns don't match any of them. Three fixes, in increasing effort:

1. **Use `DATASET_MAPPER` to point at columns explicitly** (no code edit):
   ```bash
   DATASET=your-org/qa-data \
   DATASET_MAPPER="user=question,assistant=answer" \
     ./scripts/text/launch_hf_job.sh
   ```
   Optional `system=col_c`. Format errors AND column-name typos both raise after ~10 s of dataset fetch, before `wandb.init` and model download, so no GPU time is consumed and no half-initialized W&B run is created.
2. **Extend the mapper** (~10 lines of Python): edit `_to_messages` in `scripts/text/train.py`, add an `elif` branch for your shape, return `{"messages": [...]}` with `role`/`content` pairs.
3. **Pre-process the dataset** once before training: load it via `datasets.load_dataset`, `.map()` to standard ShareGPT shape, push to your own HF dataset repo, then point `DATASET` at the new repo.

### Bad rows in your audio dataset (empty transcripts, zero-byte audio)

The audio track wires data through the `TrainingSamples` iterator you own in
`scripts/audio/train.py`, so dataset cleaning is just code: skip rows you don't want
to train on before yielding them.

```python
for row in my_rows():
    if not row["transcription"] or not row["audio"]["bytes"]:
        continue  # skip empty payloads instead of training on them
    yield [...]
```

Note the audio check inspects the nested `bytes` field; truthiness of the outer dict alone
keeps rows whose bytes are empty.

## Cluster / runtime

### Job submission succeeds but the job stays `pending` for >5 min

Check the [HF Jobs pricing page](https://huggingface.co/docs/hub/jobs-pricing) for alternate flavors and retry with a different one (`HF_FLAVOR=l40sx1` or similar). The queue clears within an hour usually.

### Job runs but exits with a cryptic `RuntimeError: NCCL ...` at startup

NCCL errors at process-init are almost always a *cascade* from one rank's Python crashing. Grep the err log for the FIRST `Traceback` / `RuntimeError` (chronologically); that's the real failure. Don't pile on NCCL env-var tweaks until the underlying Python error is identified.

### Out-of-memory mid-training

Lower the effective batch size: `BATCH_SIZE=2 GRAD_ACCUM=8 ./scripts/text/launch_hf_job.sh` (effective stays at 16). Or step down to a smaller LFM2 variant: `MODEL_ID=LiquidAI/LFM2-350M`.

---

**Still stuck?** Drop into the hackathon [Discord](https://discord.gg/WjgTAr9E) with the W&B run URL + the last 50 lines of your job's err log. Maintainers will triage.
