#!/usr/bin/env python3
"""Generate one short TTS sample from LFM2.5-Audio-1.5B (CUDA box only).

Invoked from `make smoke-audio`: loads the base model, synthesizes
~1.3 s of audio, writes `/tmp/lfm2_audio_smoke.wav` to play. Gates on
CUDA up front so a broken local setup fails in seconds with a fix hint
instead of after the model download.
"""

from __future__ import annotations

import sys
from pathlib import Path

import soundfile as sf
import torch

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B"
VOICE_PROMPT = "Perform TTS. Use the US female voice."
SAMPLE_TEXT = "Welcome to Hack the Liquid WAY."
OUTPUT_PATH = Path("/tmp/lfm2_audio_smoke.wav")
SAMPLE_RATE = 24_000  # Mimi codec; matches liquid_audio.processor.decode output


def _require_cuda() -> None:
    """liquid_audio's TTS decode is CUDA-only upstream; fail fast with
    pointers at the paths that DO work."""
    if torch.cuda.is_available():
        return
    print(
        "ERROR: liquid_audio's TTS decode requires a GPU torch backend\n"
        "(torch.cuda); this machine has none. NVIDIA CUDA and AMD ROCm\n"
        "(where torch.cuda is the HIP backend) both satisfy it.\n"
        "  - To smoke-test your general local stack: make smoke-text\n"
        "  - To smoke-test audio: use a Colab GPU runtime, or just submit:\n"
        "    the HF Jobs container is CUDA by construction and the launcher\n"
        "    gates everything else (auth, push scope, dataset schema).\n"
        "  - On the AMD Ryzen AI demo PCs: examples/on_device/README.md §2c\n"
        "    has the ROCm setup that makes the iGPU the torch device.",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    _require_cuda()
    from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor

    print(f"Loading {MODEL_ID} on cuda...")
    processor = LFM2AudioProcessor.from_pretrained(MODEL_ID, device="cuda").eval()
    model = LFM2AudioModel.from_pretrained(MODEL_ID, device="cuda").eval()

    chat = ChatState(processor)
    chat.new_turn("system")
    chat.add_text(VOICE_PROMPT)
    chat.end_turn()
    chat.new_turn("user")
    chat.add_text(SAMPLE_TEXT)
    chat.end_turn()
    chat.new_turn("assistant")

    print(f"Synthesizing: {SAMPLE_TEXT!r} ({VOICE_PROMPT})")
    audio_tokens: list[torch.Tensor] = []
    # max_new_tokens=32 keeps the smoke fast; real fine-tunes use 512+.
    for t in model.generate_sequential(
        **chat,
        max_new_tokens=32,
        audio_temperature=0.8,
        audio_top_k=64,
    ):
        if t.numel() > 1:
            audio_tokens.append(t)

    if len(audio_tokens) < 2:
        # 0-1 audio tokens = generation degenerated before a decodable chunk.
        print(
            f"ERROR: model produced {len(audio_tokens)} audio token(s); need >= 2 to decode. "
            f"This means the local stack is broken; submitting an HF Job would fail the "
            f"same way. Re-run `make validate-audio` and check the import diagnostics.",
            file=sys.stderr,
        )
        sys.exit(1)

    audio_codes = torch.stack(audio_tokens[:-1], 1).unsqueeze(0)
    waveform = processor.decode(audio_codes).cpu().squeeze().numpy()

    # 16-bit PCM mono WAV (soundfile scales the float samples and picks
    # PCM_16, its default WAV subtype); play with `afplay` / `aplay`.
    sf.write(str(OUTPUT_PATH), waveform, SAMPLE_RATE)
    duration_s = len(waveform) / SAMPLE_RATE
    print(f"✓ Wrote {OUTPUT_PATH} ({duration_s:.2f} s, {len(audio_tokens)} audio tokens)")


if __name__ == "__main__":
    main()
