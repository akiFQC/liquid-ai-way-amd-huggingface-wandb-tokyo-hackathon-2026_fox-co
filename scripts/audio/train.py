# /// script
# # Why declare deps when the repo is already locked? HF Jobs runs this in a
# # fresh container that resolves ONLY this PEP 723 block. The repo's
# # pyproject.toml / uv.lock never ship to the job. The [audio] extra in
# # pyproject.toml mirrors these pins so local `uv sync` dev stays in step.
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   # Why 2.8? liquid-audio floors torch and torchaudio at >=2.8.0 but leaves
#   # both unbounded, and their compiled extensions link against the exact peer
#   # minor. Without the <2.9 cap the resolver pairs torch 2.11 with torchaudio
#   # 2.8 and `import torchaudio` dies with a dylib symbol error in the paid
#   # container. The cap forces both to the same minor (2.8).
#   "torch>=2.8,<2.9",
#   "torchaudio>=2.8,<2.9",
#   # Pinned to a main-branch SHA, not PyPI: the 1.2.0 release still decodes
#   # dataset audio through torchcodec, which dlopens FFmpeg libs the HF Jobs
#   # container doesn't ship (OSError: libavutil.so.58). This SHA decodes via
#   # soundfile (bundled libsndfile) instead, no system FFmpeg needed.
#   # Switch back to a normal version pin at the next liquid-audio release.
#   "liquid-audio @ git+https://github.com/Liquid4All/liquid-audio@84c173b2208271dec130d0af2cfd7333a09433e1",
#   "wandb>=0.22.2",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu126", marker = "platform_system == 'Linux'" }]
# torchaudio = [{ index = "pytorch-cu126", marker = "platform_system == 'Linux'" }]
# ///
"""LFM2.5-Audio TTS fine-tune. Submitted to HF Jobs by
`scripts/audio/launch_hf_job.sh`; also runs directly on any CUDA box
(`uv run scripts/audio/train.py`).

Mirrors the upstream liquid-audio finetuning flow
(https://github.com/Liquid4All/liquid-audio#finetuning):

    yield chat messages -> preprocess to shards -> train -> save/push

**To train on your own data, rewrite `TrainingSamples.__iter__` below.**
That one iterator is the entire dataset interface; there is no
DATASET env var, no column mapping, no schema config. Yield
`[system voice prompt, user text, assistant audio bytes]` per sample
from wherever your data lives.

Env overrides (all optional; forwarded by the launcher):

    MAX_STEPS       Train steps (default 5000, the upstream recipe)
    BATCH_SIZE      Per-device batch (default 64)
    LR              Learning rate (default 1e-4)
    DATASET_SLICE   Cap example-dataset rows; 0 = all (iteration speed)
    PUSH_TO_HUB     HF repo id to publish the checkpoint to
    MODEL_ID        Base checkpoint (default LiquidAI/LFM2.5-Audio-1.5B)
    OUTPUT_DIR      Checkpoint dir (default /tmp/lfm2-audio)
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import wandb
from liquid_audio import LFM2AudioProcessor
from liquid_audio.data.dataloader import LFM2DataLoader
from liquid_audio.data.mapper import LFM2AudioChatMapper
from liquid_audio.data.preprocess import preprocess_dataset
from liquid_audio.data.types import AudioSegment, ChatMessage, TextSegment
from liquid_audio.trainer import Trainer
from liquid_audio.utils import get_model_dir

if TYPE_CHECKING:
    # Annotation-only (lazy via `from __future__ import annotations`).
    from liquid_audio.model.lfm2_audio import LFM2AudioModelOutput
    from wandb.sdk.wandb_run import Run

MODEL_ID = os.environ.get("MODEL_ID", "LiquidAI/LFM2.5-Audio-1.5B")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "5000"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
LR = float(os.environ.get("LR", "1e-4"))
DATASET_SLICE = int(os.environ.get("DATASET_SLICE", "0"))  # 0 = full dataset
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/tmp/lfm2-audio"))
PUSH_TO_HUB = os.environ.get("PUSH_TO_HUB")
# Upstream recipe constants. Edit here if your data needs longer samples.
WARMUP_STEPS = 250
CONTEXT_LENGTH = 256  # tokens per sample; longer samples are skipped


class TrainingSamples:
    """Yields one `list[ChatMessage]` per training sample.

    This class IS the dataset interface; rewrite `__iter__` for your
    own data (a HF dataset, a folder of WAVs + a CSV, an API, ...) and
    everything else in this script stays the same. For TTS, each sample
    is three turns: the system voice prompt, the text to speak, and the
    target audio as raw bytes.

    EXAMPLE: streams the Jenny TTS dataset (single English voice).

    (A class rather than a bare generator function: the upstream
    preprocessor pickles the data source for fingerprinting, and
    generator objects can't be pickled.)
    """

    # The example trains the Irish voice from Jenny; the system prompt is
    # intrinsic to that data (it's what the assistant audio answers to), so
    # it lives here rather than as an env knob. Rewrite it alongside __iter__
    # for your own data + voice.
    SYSTEM_PROMPT = "Perform TTS. Use the Irish female voice."

    def __iter__(self) -> Iterator[list[ChatMessage]]:
        from datasets import Audio, load_dataset

        ds = load_dataset("reach-vb/jenny_tts_dataset", split="train", streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))  # keep raw bytes; liquid-audio decodes
        for i, row in enumerate(ds):
            if DATASET_SLICE and i >= DATASET_SLICE:
                break
            yield [
                ChatMessage(role="system", content=[TextSegment(text=self.SYSTEM_PROMPT)]),
                ChatMessage(role="user", content=[TextSegment(text=row["transcription"])]),
                ChatMessage(role="assistant", content=[AudioSegment(audio=row["audio"]["bytes"])]),
            ]


class WandbTrainer(Trainer):
    """Upstream Trainer prints metrics to stdout but never calls
    `wandb.log()`, so WANDB_API_KEY alone does nothing; forward them."""

    def log(self, model_output: LFM2AudioModelOutput) -> None:
        super().log(model_output)
        if self.step > 0 and self.step % self.logging_interval == 0:
            wandb.log(
                {
                    "train/loss": model_output.loss.item(),
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                },
                step=self.step,
            )


def preprocess_to_shards(data_dir: Path) -> None:
    """Run every TrainingSamples chat through the mapper and write training
    shards to `data_dir`. Releases the CUDA-constructed processor + mapper
    before returning so they don't hold VRAM through training (the Trainer
    loads its own copy)."""
    print(f"[audio] Preprocessing to {data_dir}...")
    processor = LFM2AudioProcessor.from_pretrained(MODEL_ID, device="cuda").eval()
    mapper = LFM2AudioChatMapper(processor)
    preprocess_dataset(
        data=TrainingSamples(),
        output_path=str(data_dir),
        mapper=mapper,
        max_context_length=CONTEXT_LENGTH,
    )
    del processor, mapper
    torch.cuda.empty_cache()


def build_trainer(data_dir: Path) -> WandbTrainer:
    """Construct the WandbTrainer over the preprocessed shards with the
    upstream recipe (module-level config constants + the WARMUP_STEPS /
    CONTEXT_LENGTH defaults)."""
    return WandbTrainer(
        model_id=MODEL_ID,
        train_data=LFM2DataLoader(dataset_path=str(data_dir), context_length=CONTEXT_LENGTH),
        lr=LR,
        batch_size=BATCH_SIZE,
        max_steps=MAX_STEPS,
        warmup_steps=WARMUP_STEPS,
        dataloader_num_workers=8,
        logging_interval=max(1, MAX_STEPS // 100),
        save_interval=max(100, MAX_STEPS // 10),
        output_dir=str(OUTPUT_DIR),
    )


def assemble_checkpoint(final_dir: Path) -> None:
    """Make `final_dir` a self-contained, loadable checkpoint. The Trainer
    writes ONLY model.safetensors there; copy the base snapshot's other
    files alongside (config, tokenizer, Mimi codec, detokenizer) so
    from_pretrained can load it from a single dir or pushed Hub repo."""
    if not (final_dir / "model.safetensors").exists():
        raise RuntimeError(
            f"Training ended but {final_dir} has no model.safetensors; "
            f"check the logs above for a save failure."
        )
    base_dir = get_model_dir(MODEL_ID)  # cache hit, downloaded at training start
    for entry in base_dir.iterdir():
        # Top-level model.safetensors is the only file training replaces;
        # audio_detokenizer/model.safetensors (nested) must still be copied.
        if entry.name == "model.safetensors" or entry.name.startswith("."):
            continue
        if entry.is_dir():
            shutil.copytree(entry, final_dir / entry.name, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, final_dir / entry.name)
    print(f"[audio] Checkpoint at: {final_dir}")


def push_checkpoint(folder: Path, repo_id: str, run: Run) -> None:
    """Upload the checkpoint to the Hub in one commit and stamp the commit
    SHA into the W&B run summary, which `verify_run --require-hf-repo` reads
    to prove THIS run produced what's on the Hub."""
    from huggingface_hub import HfApi

    print(f"[audio] Pushing to https://huggingface.co/{repo_id} ...")
    api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        commit_message="hf-jobs trained checkpoint",
    )
    run.summary["hf_revision"] = commit.oid
    run.summary["hf_repo"] = repo_id
    print(f"[audio] Push complete. Hub revision: {commit.oid}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This fine-tune requires CUDA. Re-submit with a GPU flavor "
            "(HF_FLAVOR=a100-large or another at "
            "https://huggingface.co/docs/hub/jobs-pricing)."
        )

    # Run as a context manager so an exception anywhere below finalizes the
    # run as Crashed. A bare wandb.finish() at the end would be skipped if
    # training raised, leaving the W&B run stuck Running.
    with wandb.init(
        project=os.environ.get("WANDB_PROJECT", "hack-the-liquid-way"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=os.environ.get("WANDB_RUN_NAME"),
        tags=["audio", "lfm2.5-audio-1.5b", "hf-jobs"],
        config={
            "model": MODEL_ID,
            "max_steps": MAX_STEPS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "dataset_slice": DATASET_SLICE,
        },
    ) as run:
        print(f"[audio] W&B run URL: {run.url}")

        data_dir = OUTPUT_DIR / "preprocessed"
        preprocess_to_shards(data_dir)

        trainer = build_trainer(data_dir)
        trainer.train()

        assemble_checkpoint(OUTPUT_DIR / "final")
        if PUSH_TO_HUB:
            push_checkpoint(OUTPUT_DIR / "final", PUSH_TO_HUB, run)

    print("[audio] DONE")


if __name__ == "__main__":
    main()
