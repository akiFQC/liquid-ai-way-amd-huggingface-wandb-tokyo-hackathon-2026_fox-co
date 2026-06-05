"""Gradio chat UI for your fine-tuned LFM2.

Run locally (`--with gradio` adds gradio ephemerally; a plain
`uv pip install gradio` would be stripped again by uv run's sync):

    MODEL_ID=your-username/your-finetune \\
        uv run --with gradio python examples/demo/text_chat.py

Defaults to LiquidAI/LFM2-350M for a quick spin without a fine-tune.

To deploy to a HuggingFace Space, push this file + a requirements.txt
(transformers, torch, gradio) to a new Space. Set MODEL_ID as a
Space secret.
"""

from __future__ import annotations

import os

import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

MODEL_ID = os.environ.get("MODEL_ID", "LiquidAI/LFM2-350M")


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


def respond(
    message: str,
    history: list[tuple[str, str]],
    system_prompt: str,
    tok: PreTrainedTokenizerBase,
    model: PreTrainedModel,
) -> str:
    """Canonical LFM2 generation per https://huggingface.co/LiquidAI/LFM2-350M.

    `tok` and `model` are passed in (rather than module-level) so this function
    can be unit-tested with mocks; no 1.4 GB model download just to verify the
    chat-template assembly is correct.
    """
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    # The three `ty: ignore`s below are all transformers stub gaps, not real
    # errors (the code runs; tests cover it): apply_chat_template is overloaded
    # on (tokenize, return_dict) but its stub returns the whole union, so ty
    # can't see the `.to()`-able BatchEncoding; generate() lives on
    # GenerationMixin, which ty reaches via nn.Module.__getattr__ (Tensor |
    # Module); and BatchEncoding[...] is stub-typed `Any | Encoding`.
    inputs = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to(model.device)  # ty: ignore[unresolved-attribute]
    gen = model.generate(  # ty: ignore[call-non-callable]
        **inputs,
        do_sample=True,
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05,
        max_new_tokens=512,
    )
    new_tokens = gen[0][inputs["input_ids"].shape[1] :]  # ty: ignore[unresolved-attribute]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def build_demo(tok: PreTrainedTokenizerBase, model: PreTrainedModel) -> gr.ChatInterface:
    """Assemble the Gradio ChatInterface bound to a specific tokenizer + model.

    Pulled out of module scope so the UI structure is testable without a real
    model load.
    """

    def _respond(message: str, history: list[tuple[str, str]], system_prompt: str) -> str:
        return respond(message, history, system_prompt, tok, model)

    return gr.ChatInterface(
        _respond,
        additional_inputs=[
            gr.Textbox(
                value="You are a helpful assistant trained by Liquid AI.",
                label="System prompt",
            ),
        ],
        title=f"💧 {MODEL_ID}",
        description=(
            "Canonical LFM2 sampling: temperature=0.3, min_p=0.15, repetition_penalty=1.05."
        ),
    )


def main() -> None:  # pragma: no cover  (model load + UI launch; covered by manual smoke)
    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    device = _device()
    model = (
        AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=_dtype_for_device(device))
        .eval()
        .to(device)
    )
    build_demo(tok, model).launch()


if __name__ == "__main__":
    main()
