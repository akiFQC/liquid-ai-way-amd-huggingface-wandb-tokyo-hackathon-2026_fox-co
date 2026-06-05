#!/usr/bin/env python3
"""Compare a base LFM2 against your fine-tune on a held-out prompt set.

Runs the same prompts through both models with the canonical LFM2 sampling
recipe and prints a side-by-side table. Optionally logs every call to W&B +
Weave so the comparison is reproducible from the dashboard.

Usage:
    python scripts/run_eval.py \\
        --base LiquidAI/LFM2-350M \\
        --finetune your-username/your-finetune \\
        [--prompts path/to/prompts.txt] \\
        [--wandb]

`prompts.txt` is one prompt per line. Omit the flag to use the default
4-prompt Tokyo-themed set built in.
"""

from __future__ import annotations

import argparse
import contextlib
import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

DEFAULT_PROMPTS = [
    "Tokyo is famous for what cuisine and culture?",
    "Translate to Japanese: hello, how are you?",
    "Briefly: what is a small language model good for?",
    "What is the LFM2 architecture?",
]


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_for_device(device: str) -> torch.dtype:
    """Pick a sane dtype per device. PyTorch's MPS backend doesn't support
    bfloat16 (TypeError: BFloat16 is not supported), so the Mac path
    falls back to float16. CPU stays at float32; both lower-precision
    options are slow without hardware accumulators on x86 / Apple Silicon
    CPU, and float32 is the safest default."""
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def load(model_id: str) -> tuple:
    print(f"Loading {model_id}...")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = _device()
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=_dtype_for_device(device),
        )
        .eval()
        .to(device)
    )
    return tok, model


def generate(
    tok: PreTrainedTokenizerBase, model: PreTrainedModel, prompt: str, max_new_tokens: int = 256
) -> str:
    """Canonical LFM2 sampling per https://huggingface.co/LiquidAI/LFM2-350M."""
    # The three `ty: ignore`s are transformers stub gaps, not real errors (the
    # code runs; tests cover it): apply_chat_template's overload union hides the
    # `.to()`-able BatchEncoding, generate() lives on GenerationMixin (ty reaches
    # it via nn.Module.__getattr__ -> Tensor | Module), and BatchEncoding[...] is
    # stub-typed `Any | Encoding`.
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
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
        max_new_tokens=max_new_tokens,
    )
    new_tokens = gen[0][inputs["input_ids"].shape[1] :]  # ty: ignore[unresolved-attribute]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def print_markdown_table(rows: list[dict[str, str]]) -> None:
    """Print the base-vs-finetune rows as a markdown table (cells truncated
    and pipe/newline-escaped) the demo deck can paste."""
    print("\n## Markdown comparison\n")
    print("| Prompt | Base | Fine-tune |")
    print("|---|---|---|")
    for r in rows:
        prompt = r["prompt"][:60].replace("|", "\\|")
        base = r["base"][:120].replace("\n", " ").replace("|", "\\|")
        ft = r["finetune"][:120].replace("\n", " ").replace("|", "\\|")
        print(f"| {prompt} | {base} | {ft} |")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="LiquidAI/LFM2-350M")
    parser.add_argument("--finetune", required=True, help="HF Hub repo id or local checkpoint path")
    parser.add_argument("--prompts", help="Path to a file with one prompt per line")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--wandb", action="store_true", help="Log every comparison to W&B + Weave")
    args = parser.parse_args()

    if args.prompts:
        with open(args.prompts) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = DEFAULT_PROMPTS

    def compare(prompt: str, base: str, ft: str) -> dict[str, str]:
        return {"prompt": prompt, "base": base, "finetune": ft}

    # With --wandb: wrap `compare` as a traced @weave.op, and stamp every call
    # with this run's id so the Weave UI can filter the eval's traces in a
    # shared hackathon project. The two `ty: ignore`s are not real errors:
    # weave.op() returns an Op (callable, same signature) that ty won't accept
    # as a rebind of the `compare` function; and wandb.run is set the moment
    # wandb.init() runs above it, but ty can't narrow the module-level global.
    trace_attrs = contextlib.nullcontext()
    if args.wandb:
        import wandb
        import weave

        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "hack-the-liquid-way"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=os.environ.get("WANDB_RUN_NAME", "run_eval"),
            tags=["eval", "compare"],
            config={"base": args.base, "finetune": args.finetune, "n_prompts": len(prompts)},
        )
        weave.init(
            f"{os.environ.get('WANDB_ENTITY', '')}/{os.environ.get('WANDB_PROJECT', 'hack-the-liquid-way')}".lstrip(
                "/"
            )
        )
        compare = weave.op()(compare)  # ty: ignore[invalid-assignment]
        trace_attrs = weave.attributes({"wandb_run_id": wandb.run.id})  # ty: ignore[unresolved-attribute]

    base_tok, base_model = load(args.base)
    ft_tok, ft_model = load(args.finetune)

    rows: list[dict[str, str]] = []
    with trace_attrs:
        for prompt in prompts:
            base_out = generate(base_tok, base_model, prompt, args.max_new_tokens)
            ft_out = generate(ft_tok, ft_model, prompt, args.max_new_tokens)
            rows.append(compare(prompt=prompt, base=base_out, ft=ft_out))
            print(f"\n=== {prompt} ===")
            print(f"BASE      ({args.base}): {base_out}")
            print(f"FINETUNE  ({args.finetune}): {ft_out}")

    print_markdown_table(rows)

    if args.wandb:
        import wandb

        wandb.log(
            {
                "comparison_table": wandb.Table(
                    columns=["prompt", "base", "finetune"],
                    data=[[r["prompt"], r["base"], r["finetune"]] for r in rows],
                )
            }
        )
        wandb.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
