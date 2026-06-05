# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch>=2.8,<2.13",
#   "transformers>=4.45",
#   "trl>=0.13",
#   "datasets>=4.0",
#   "wandb>=0.22.2",
#   "accelerate>=1.0",
#   "peft>=0.17",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
#
# [tool.uv.sources]
# torch = [
#   { index = "pytorch-cu126", marker = "platform_system == 'Linux'" },
# ]
# ///
"""LFM2 text LoRA fine-tune. Submitted to HF Jobs by
`scripts/text/launch_hf_job.sh`; also runs directly on any CUDA box
(`uv run scripts/text/train.py`).

Stack: TRL `SFTTrainer` + PEFT LoRA. The LoRA recipe follows the Liquid
AI docs (https://docs.liquid.ai/lfm/fine-tuning/unsloth): r=16,
alpha=32, dropout=0, all linear target modules. To hear the result,
compare base vs fine-tune afterwards with `scripts/run_eval.py`.

Env overrides (all optional; forwarded by the launcher):

    DATASET         HF dataset id (default mlabonne/FineTome-100k-dedup).
                    Five common chat shapes are auto-detected; for other
                    column layouts set DATASET_MAPPER or extend
                    `_to_messages()` below.
    DATASET_MAPPER  Explicit column-to-role mapping, e.g.
                    'user=question,assistant=answer[,system=col]'
    DATASET_SLICE   Row cap (default 1024; 0 = full dataset)
    MAX_STEPS       Train steps (default 200)
    BATCH_SIZE      Per-device batch (default 4)
    LR              Learning rate (default 2e-4)
    PUSH_TO_HUB     HF repo id to publish the merged checkpoint to
    MODEL_ID        Base checkpoint (default LiquidAI/LFM2-350M).
                    LFM2 family + LFM2-VL only; other architectures need
                    a different LoRA target_modules list below.
    OUTPUT_DIR      Checkpoint dir (default /tmp/lfm2-text)

(The two mapper helpers stay at module level and import-free so the
launcher's dataset preflight can import them without pulling torch; the
torch / trl / peft imports live inside the phase functions that use them.)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Annotation-only imports (lazy via `from __future__ import annotations`,
    # so no runtime cost and no break to this module's import-lightness).
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    from trl import SFTTrainer
    from wandb.sdk.wandb_run import Run


def _parse_dataset_mapper(raw: str | None) -> dict[str, str] | None:
    """Parse the DATASET_MAPPER env var into a {role: column} dict.

    Format: 'user=col_a,assistant=col_b[,system=col_c]'. Returns None when
    `raw` is empty / unset. Raises ValueError on malformed input (missing
    `=`, unknown role, missing required `user` / `assistant`) so the
    participant gets a fail-fast error before training launches.
    """
    if not raw:
        return None
    mapper: dict[str, str] = {}
    for pair in raw.split(","):
        if not pair.strip():
            continue
        if "=" not in pair:
            raise ValueError(f"Malformed DATASET_MAPPER pair: {pair!r} (expected 'role=column')")
        role, _, col = pair.partition("=")
        role, col = role.strip(), col.strip()
        if role not in ("user", "assistant", "system"):
            raise ValueError(
                f"Unknown role {role!r} in DATASET_MAPPER (allowed: user, assistant, system)"
            )
        mapper[role] = col
    if "user" not in mapper or "assistant" not in mapper:
        raise ValueError(
            f"DATASET_MAPPER must include both 'user' and 'assistant' roles. Got: {sorted(mapper)}"
        )
    return mapper


def _to_messages(example: dict, mapper: dict[str, str] | None = None) -> dict:
    """Map a dataset row to the chat-format `{"messages": [...]}` SFTTrainer
    expects. If `mapper` is provided (from DATASET_MAPPER env), it wins;
    explicit user-supplied column mapping bypasses auto-detection. Otherwise
    five common shapes seen on the Hub are tried in order; extend here for
    any dataset whose column layout isn't covered."""
    # 0. Explicit mapper; user told us which columns map to which roles.
    if mapper:
        # Every mapped role must point at a column that exists in the row,
        # so a typo'd column name raises instead of silently dropping a turn.
        missing = [col for col in mapper.values() if col not in example]
        if missing:
            raise ValueError(
                f"DATASET_MAPPER references columns not in dataset row. "
                f"Missing: {missing}; mapper: {mapper}; "
                f"row columns: {sorted(example.keys())}"
            )
        messages: list[dict[str, str]] = []
        if "system" in mapper and example[mapper["system"]]:
            messages.append({"role": "system", "content": example[mapper["system"]]})
        messages.append({"role": "user", "content": example[mapper["user"]]})
        messages.append({"role": "assistant", "content": example[mapper["assistant"]]})
        return {"messages": messages}
    # 1. Native ShareGPT messages (FineTome-100k, Aratako synthetic, etc.)
    if "messages" in example and isinstance(example["messages"], list):
        return {"messages": example["messages"]}
    # 2. ShareGPT conversations (FineTome-dedup, shi3z/ja_conv_wikipedia, shisa-v2)
    if "conversations" in example and isinstance(example["conversations"], list):
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        return {
            "messages": [
                {"role": role_map.get(turn["from"], turn["from"]), "content": turn["value"]}
                for turn in example["conversations"]
            ]
        }
    # 3. Alpaca-style (instruction + optional input -> output). Covers
    #    fujiki/japanese_alpaca_data, izumi-lab/llm-japanese-dataset.
    if "instruction" in example and "output" in example:
        user_text = example["instruction"]
        if example.get("input"):
            user_text = f"{user_text}\n\n{example['input']}"
        return {
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": example["output"]},
            ]
        }
    # 4. Dolly-style (instruction + optional context -> response). Covers
    #    kunishou/databricks-dolly-15k-ja, llm-jp/databricks-dolly-15k-ja.
    if "instruction" in example and "response" in example:
        user_text = example["instruction"]
        if example.get("context"):
            user_text = f"{user_text}\n\n{example['context']}"
        return {
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": example["response"]},
            ]
        }
    # 5. Aya-style (inputs -> targets). Covers CohereForAI/aya_dataset.
    if "inputs" in example and "targets" in example:
        return {
            "messages": [
                {"role": "user", "content": example["inputs"]},
                {"role": "assistant", "content": example["targets"]},
            ]
        }
    raise ValueError(
        f"Dataset row has no recognized shape. Got columns: {sorted(example.keys())}. "
        f"Either set DATASET_MAPPER='user=<col>,assistant=<col>' to map columns "
        f"explicitly, or extend _to_messages() to handle the dataset's column layout."
    )


def load_training_dataset(
    dataset_name: str, dataset_slice: int, mapper: dict[str, str] | None
) -> Dataset:
    """Load DATASET and map every row to chat format. Run BEFORE wandb.init +
    the model download so a DATASET typo / column mismatch fails in seconds
    with no orphan W&B run left behind."""
    from datasets import load_dataset

    # `train[:N]` fetches only N rows; `.select(range(N))` would download
    # the whole split first.
    split_spec = f"train[:{dataset_slice}]" if dataset_slice > 0 else "train"
    ds = load_dataset(dataset_name, split=split_spec)
    ds = ds.map(lambda ex: _to_messages(ex, mapper), remove_columns=ds.column_names)
    print(f"[text] Prepared {len(ds)} samples.")
    return ds


def load_base_model(model_id: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the tokenizer + model ready for training. Sets pad_token (LFM2
    tokenizers leave it unset, which crashes batching) and bf16 weights with
    an automatic device map."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,  # transformers 5.x; torch_dtype is deprecated
        device_map="auto",
    )
    return model, tok


def build_trainer(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    dataset: Dataset,
    *,
    max_steps: int,
    batch_size: int,
    lr: float,
    output_dir: str,
) -> SFTTrainer:
    """Wrap the model in the canonical LFM2 LoRA recipe and return a
    ready-to-train SFTTrainer.

    LoRA per https://docs.liquid.ai/lfm/fine-tuning/unsloth: r=16, alpha=32,
    dropout=0, all-linear targets (the target_modules names assume
    LFM2-family attention/MLP layers)."""
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
        bias="none",
    )
    cfg = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,  # effective batch = batch_size * 4
        max_steps=max_steps,
        learning_rate=lr,
        warmup_steps=max(10, max_steps // 20),  # 5% of training, floor 10
        lr_scheduler_type="linear",
        optim="adamw_torch",
        weight_decay=0.01,
        logging_steps=max(1, max_steps // 50),  # log every 2% of training, floor 1
        report_to=["wandb"],
        bf16=True,
        # Periodic adapter checkpoints so a mid-train OOM / preemption
        # doesn't discard all progress. Keep the latest 2 to bound disk.
        save_strategy="steps",
        save_steps=max(50, max_steps // 10),
        save_total_limit=2,
        max_length=1024,
        packing=False,
    )
    return SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        processing_class=tok,
        peft_config=peft_config,
    )


def save_merged_checkpoint(
    trainer: SFTTrainer, tok: PreTrainedTokenizerBase, output_dir: str
) -> None:
    """Save the deployable checkpoint. Writes the LoRA adapter first as
    insurance: if merge_and_unload fails, the trained weights are still
    recoverable from `<output_dir>-adapter`, then merges the adapter into
    the base weights and saves the standalone model."""
    adapter_dir = f"{output_dir}-adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)

    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    print(f"[text] Merged checkpoint saved to {output_dir}")


def push_checkpoint(folder: str, repo_id: str, run: Run) -> None:
    """Upload a finished checkpoint to the Hub in one commit (a transient
    failure can't leave a half-published repo) and stamp the commit SHA into
    the W&B run summary, which `verify_run --require-hf-repo` reads to prove
    THIS run produced what's on the Hub."""
    from huggingface_hub import HfApi

    print(f"[text] Pushing to https://huggingface.co/{repo_id} ...")
    api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True)
    commit = api.upload_folder(
        folder_path=folder,
        repo_id=repo_id,
        commit_message="hf-jobs trained merged checkpoint",
    )
    run.summary["hf_revision"] = commit.oid
    run.summary["hf_repo"] = repo_id
    print(f"[text] Push complete. Hub revision: {commit.oid}")


def main() -> None:
    import torch
    import wandb

    model_id = os.environ.get("MODEL_ID", "LiquidAI/LFM2-350M")
    dataset_name = os.environ.get("DATASET", "mlabonne/FineTome-100k-dedup")
    dataset_slice = int(os.environ.get("DATASET_SLICE", "1024"))  # 0 = full dataset
    max_steps = int(os.environ.get("MAX_STEPS", "200"))
    batch_size = int(os.environ.get("BATCH_SIZE", "4"))
    lr = float(os.environ.get("LR", "2e-4"))
    output_dir = os.environ.get("OUTPUT_DIR", "/tmp/lfm2-text")
    push_to_hub = os.environ.get("PUSH_TO_HUB")
    mapper = _parse_dataset_mapper(os.environ.get("DATASET_MAPPER"))

    print(
        f"[text] torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"model={model_id} dataset={dataset_name} slice={dataset_slice} "
        f"max_steps={max_steps} bs={batch_size} lr={lr}"
    )

    ds = load_training_dataset(dataset_name, dataset_slice, mapper)

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "hack-the-liquid-way"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=os.environ.get("WANDB_RUN_NAME"),
        tags=["text", "lfm2", "lora", "hf-jobs"],
        config={
            "model": model_id,
            "dataset": dataset_name,
            "dataset_slice": dataset_slice,
            "dataset_mapper": mapper,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "lr": lr,
        },
    )
    print(f"[text] W&B run URL: {run.url}")

    model, tok = load_base_model(model_id)
    trainer = build_trainer(
        model, tok, ds, max_steps=max_steps, batch_size=batch_size, lr=lr, output_dir=output_dir
    )
    print("[text] Starting training...")
    out = trainer.train()
    print(f"[text] FINAL TRAIN LOSS: {out.training_loss}")

    save_merged_checkpoint(trainer, tok, output_dir)
    if push_to_hub:
        push_checkpoint(output_dir, push_to_hub, run)

    wandb.finish()
    print("[text] DONE")


if __name__ == "__main__":
    main()
