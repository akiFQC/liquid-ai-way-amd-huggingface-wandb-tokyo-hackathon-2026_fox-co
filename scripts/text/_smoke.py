#!/usr/bin/env python3
"""Generate one short sample from LFM2-350M on whatever device is available.

Invoked from `make smoke-text` to confirm transformers + torch can load the
model on the participant's machine without a W&B / Hub round-trip. Uses
the canonical LFM2 sampling recipe so the output should be a coherent
3-word answer; doom-loops here mean the recipe is broken.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "LiquidAI/LFM2-350M"


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_for_device(device: str) -> torch.dtype:
    """Pick dtype per device; PyTorch MPS doesn't support bfloat16."""
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def main() -> None:
    device = _device()
    print(f"Loading {MODEL_ID} on {device}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = (
        AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=_dtype_for_device(device))
        .eval()
        .to(device)
    )
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": "Hello in 3 words"}],
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to(device)
    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05,
        max_new_tokens=16,
    )
    print(tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True))


if __name__ == "__main__":
    main()
