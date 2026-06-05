# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   # torch + torchaudio minors MUST match (see scripts/audio/train.py
#   # for the full ABI-mismatch story).
#   "torch>=2.8,<2.9",
#   "torchaudio>=2.8,<2.9",
#   # torchcodec-free SHA: decodes via soundfile, no system FFmpeg (see the
#   # [audio] extra in pyproject.toml for the full story). Same pin as
#   # scripts/audio/train.py.
#   "liquid-audio @ git+https://github.com/Liquid4All/liquid-audio@84c173b2208271dec130d0af2cfd7333a09433e1",
#   "soundfile>=0.12",  # write_wav() encodes via soundfile
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
"""Compare a base LFM2.5-Audio TTS against your fine-tune on a fixed prompt set.

Synthesizes the same text prompts through both models with the canonical
TTS sampling recipe, writes WAV pairs to `./eval_audio/`, and prints a
side-by-side markdown table the demo deck can screenshot. Optionally
logs the audio pairs as `wandb.Audio` entries so the comparison is
reproducible from the W&B dashboard.

REQUIRES CUDA: liquid_audio's audio detokenizer is CUDA-only upstream,
so this can't run on Apple Silicon / CPU-only machines. The cheapest
path is an HF Jobs submission on the `l4x1` flavor ($0.80/h; this eval
finishes in ~10 min ≈ $0.15):

    hf jobs uv run --flavor l4x1 --timeout 30m \\
        --secrets HF_TOKEN --secrets WANDB_API_KEY \\
        --env WANDB_ENTITY=<entity> --env WANDB_PROJECT=<project> \\
        --env FINETUNE=<your-username>/<your-audio-finetune> \\
        scripts/run_eval_audio.py

(The script reads FINETUNE from env when no --finetune flag is given,
because `hf jobs uv run` can't pass script-level CLI flags.)

On a CUDA box / Colab GPU it also runs directly:

    python scripts/run_eval_audio.py \\
        --finetune your-username/your-audio-finetune [--wandb]

`--prompts path.txt` swaps the built-in 3-prompt set for one
text-to-synthesize per line.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING

import soundfile as sf
import torch

if TYPE_CHECKING:
    # Annotation-only (lazy via `from __future__ import annotations`).
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

DEFAULT_PROMPTS = [
    "Welcome to Hack the Liquid WAY.",
    "The quick brown fox jumps over the lazy dog.",
    "Tokyo is the capital of Japan and host of this hackathon.",
]
DEFAULT_VOICE_PROMPT = "Perform TTS. Use the Irish female voice."
SAMPLE_RATE = 24_000  # Mimi codec
OUTPUT_DIR = Path("eval_audio")


def _require_cuda() -> None:
    """liquid_audio's TTS decode is CUDA-only upstream; fail fast here
    rather than after minutes of model loading + generation."""
    if torch.cuda.is_available():
        return
    raise SystemExit(
        "ERROR: liquid_audio's TTS decode requires a GPU torch backend "
        "(torch.cuda: NVIDIA CUDA or AMD ROCm both satisfy it). Run this "
        "script on a GPU box, e.g. submit it to HF Jobs (`hf jobs uv run "
        "--flavor l4x1 ...`) or use a Colab GPU runtime. Apple Silicon / "
        "CPU-only machines cannot decode audio locally."
    )


def load(model_id: str) -> tuple[LFM2AudioProcessor, LFM2AudioModel]:
    """Load the LFM2.5-Audio processor + model pair (CUDA-only; see
    `_require_cuda`)."""
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

    # liquid_audio treats a str as a Hub repo id (snapshot_download) and
    # only a Path object as a local dir; normalize so --finetune accepts
    # either a pushed repo or a local OUTPUT_DIR/final checkpoint.
    source = Path(model_id) if Path(model_id).exists() else model_id
    print(f"Loading {model_id}...")
    processor = LFM2AudioProcessor.from_pretrained(source, device="cuda").eval()
    model = LFM2AudioModel.from_pretrained(source, device="cuda").eval()
    return processor, model


def synthesize(
    processor: LFM2AudioProcessor, model: LFM2AudioModel, text: str, voice_prompt: str
) -> torch.Tensor:
    """Canonical LFM2.5-Audio TTS sampling per liquid-audio README:
    audio_temperature=0.8, audio_top_k=64 (sequential generation)."""
    from liquid_audio import ChatState

    chat = ChatState(processor)
    chat.new_turn("system")
    chat.add_text(voice_prompt)
    chat.end_turn()
    chat.new_turn("user")
    chat.add_text(text)
    chat.end_turn()
    chat.new_turn("assistant")

    audio_tokens: list[torch.Tensor] = []
    for t in model.generate_sequential(
        **chat,
        max_new_tokens=256,
        audio_temperature=0.8,
        audio_top_k=64,
    ):
        if t.numel() > 1:
            audio_tokens.append(t)

    if len(audio_tokens) < 2:
        # 0-1 audio tokens = generation degenerated before a decodable chunk.
        raise RuntimeError(
            f"Model produced {len(audio_tokens)} audio token(s); need >= 2 to decode. "
            f"The fine-tune may not have learned the voice prompt {voice_prompt!r}."
        )
    audio_codes = torch.stack(audio_tokens[:-1], 1).unsqueeze(0)
    return processor.decode(audio_codes).cpu().squeeze()


def write_wav(path: Path, waveform: torch.Tensor) -> None:
    """Save a 1-D float waveform as a 16-bit PCM mono WAV at 24 kHz.

    soundfile scales the float samples and picks PCM_16 (its default
    subtype for WAV) on its own, so the manual int16 conversion the kit
    used to do is unnecessary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform.numpy(), SAMPLE_RATE)


def print_markdown_table(rows: list[dict[str, str]]) -> None:
    """Print the base-vs-finetune WAV paths as a markdown table the demo
    deck can paste (cells pipe-escaped)."""
    print("\n## Markdown comparison\n")
    print("| Prompt | Base WAV | Fine-tune WAV |")
    print("|---|---|---|")
    for r in rows:
        prompt = r["prompt"][:60].replace("|", "\\|")
        base = r["base_wav"].replace("|", "\\|")
        ft = r["ft_wav"].replace("|", "\\|")
        print(f"| {prompt} | `{base}` | `{ft}` |")


def log_audio_comparison(rows: list[dict[str, str]]) -> None:
    """Log each prompt's base/finetune pair as side-by-side wandb.Audio
    entries (inline playable players in the dashboard) plus a summary
    comparison table, then finish the run."""
    import wandb

    for i, r in enumerate(rows):
        wandb.log(
            {
                f"prompt_{i:02d}/base": wandb.Audio(r["base_wav"], sample_rate=SAMPLE_RATE),
                f"prompt_{i:02d}/finetune": wandb.Audio(r["ft_wav"], sample_rate=SAMPLE_RATE),
                f"prompt_{i:02d}/text": wandb.Html(f"<p>{r['prompt']}</p>"),
            }
        )
    wandb.log(
        {
            "comparison_table": wandb.Table(
                columns=["prompt", "base_wav", "finetune_wav"],
                data=[[r["prompt"], r["base_wav"], r["ft_wav"]] for r in rows],
            )
        }
    )
    wandb.finish()


def main() -> int:
    _require_cuda()
    parser = argparse.ArgumentParser()
    # Env-var fallbacks (FINETUNE, BASE, WANDB=1) exist because
    # `hf jobs uv run <script>` can't pass script-level CLI flags;
    # the launcher-style `--env` forwarding is the only channel.
    parser.add_argument("--base", default=os.environ.get("BASE", "LiquidAI/LFM2.5-Audio-1.5B"))
    parser.add_argument(
        "--finetune",
        default=os.environ.get("FINETUNE") or None,
        help="HF Hub repo id or local checkpoint path (env fallback: FINETUNE)",
    )
    parser.add_argument("--prompts", help="Path to a file with one text-to-synthesize per line")
    parser.add_argument(
        "--voice-prompt",
        default=DEFAULT_VOICE_PROMPT,
        help='Voice system prompt (e.g. "Perform TTS. Use the Irish female voice.")',
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log every audio pair to W&B (env fallback: WANDB=1)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Local directory for WAV pairs (default: ./eval_audio)",
    )
    args = parser.parse_args()
    if not args.finetune:
        parser.error("--finetune (or env FINETUNE) is required")
    use_wandb = args.wandb or os.environ.get("WANDB", "") == "1"

    if args.prompts:
        with open(args.prompts) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = DEFAULT_PROMPTS

    output_dir = Path(args.output_dir)

    if use_wandb:
        import wandb

        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "hack-the-liquid-way"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=os.environ.get("WANDB_RUN_NAME", "run_eval_audio"),
            tags=["eval", "compare", "audio"],
            config={
                "base": args.base,
                "finetune": args.finetune,
                "voice_prompt": args.voice_prompt,
                "n_prompts": len(prompts),
            },
        )

    base_proc, base_model = load(args.base)
    ft_proc, ft_model = load(args.finetune)

    rows: list[dict[str, str]] = []
    for i, prompt in enumerate(prompts):
        print(f"\n=== {prompt} ===")
        base_wave = synthesize(base_proc, base_model, prompt, args.voice_prompt)
        ft_wave = synthesize(ft_proc, ft_model, prompt, args.voice_prompt)
        base_path = output_dir / f"base_{i:02d}.wav"
        ft_path = output_dir / f"finetune_{i:02d}.wav"
        write_wav(base_path, base_wave)
        write_wav(ft_path, ft_wave)
        print(f"BASE      ({args.base}): {base_path}")
        print(f"FINETUNE  ({args.finetune}): {ft_path}")
        rows.append({"prompt": prompt, "base_wav": str(base_path), "ft_wav": str(ft_path)})

    print_markdown_table(rows)
    if use_wandb:
        log_audio_comparison(rows)

    print(f"\n✓ WAV pairs written to {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
