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
"""fox-co PII抽出タスク向け LFM2 LoRA fine-tune スクリプト。

scripts/text/train.py をベースに、以下を追加:
  - chat_template.jinja をトークナイザーに注入
  - 学習後に JSON parse 成功率・カテゴリ別エンティティF1 を評価してW&Bに記録

NOTE: HF Jobs に提出する場合は chat_template.jinja をこのスクリプト内に
インライン化すること（HF Jobs は単一ファイルをアップロードするため）。

Env overrides (all optional unless noted):

    DATASET         HF dataset id (必須). 一般的なSFT形式を自動検出:
                    messages, conversations, instruction+output, instruction+response,
                    inputs+targets。その他は DATASET_MAPPER か _to_messages() を拡張。
    DATASET_MAPPER  明示的な列マッピング: 'user=q,assistant=a[,system=s]'
    DATASET_SLICE   行数上限 (default 1024; 0 = 全件)
    EVAL_SPLIT_RATIO train から切り出す eval 比率 (default 0.1)
    EVAL_SAMPLES    評価サンプル数 (default 50; 0 = スキップ)
    SKIP_TRAINING   1 に設定すると学習をスキップしてevalのみ実行 (default 0)
    MAX_STEPS       学習ステップ数 (default 200)
    BATCH_SIZE      per-device batch size (default 4)
    LR              learning rate (default 2e-4)
    PUSH_TO_HUB     マージ済みチェックポイントの HF repo id
    MODEL_ID        ベースモデル (default LiquidAI/LFM2-350M)
    OUTPUT_DIR      チェックポイント保存先 (default /tmp/lfm2-fox-co)
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    from trl import SFTTrainer
    from wandb.sdk.wandb_run import Run


# ---------------------------------------------------------------------------
# Eval utilities (inlined from eval.py for HF Jobs single-file compatibility)
# ---------------------------------------------------------------------------

PII_CATEGORIES: list[str] = [
    "address",
    "company_name",
    "email_address",
    "human_name",
    "phone_number",
    "account_identifier",
    "network_identifier",
    "system_config",
    "project_info",
    "financial_info",
    "transaction_id",
]


@dataclass
class CategoryResult:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


@dataclass
class EvalResult:
    n_samples: int
    json_parse_rate: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    per_category: dict[str, CategoryResult] = field(default_factory=dict)


def run_eval(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    eval_ds: Dataset,
    n_samples: int,
) -> EvalResult:
    """Run PII extraction eval. Pure: returns EvalResult, no side effects.

    Canonical LFM2 generation recipe is used (temperature=0.3, min_p=0.15,
    repetition_penalty=1.05). Entity matching is exact string match.
    """
    import json

    import torch

    # CRITICAL: training leaves dropout active; must call eval() before generate().
    model.eval()

    actual = min(n_samples, len(eval_ds))
    if actual == 0:
        return EvalResult(
            n_samples=0,
            json_parse_rate=0.0,
            micro_precision=0.0,
            micro_recall=0.0,
            micro_f1=0.0,
        )

    samples = eval_ds.select(range(actual))
    json_ok = 0
    cat_stats: dict[str, dict[str, int]] = {
        k: {"tp": 0, "fp": 0, "fn": 0} for k in PII_CATEGORIES
    }

    for row in samples:
        msgs = row["messages"]
        prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
        gold_str = next((m["content"] for m in msgs if m["role"] == "assistant"), "{}")

        inputs = tok.apply_chat_template(
            prompt_msgs,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        ).to(model.device)

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.3,
                min_p=0.15,
                repetition_penalty=1.05,
                max_new_tokens=512,
            )

        new_tokens = gen[0][inputs["input_ids"].shape[1]:]
        raw_output = tok.decode(new_tokens, skip_special_tokens=True).strip()

        try:
            pred = json.loads(raw_output)
            if not isinstance(pred, dict):
                raise ValueError
            json_ok += 1
        except (json.JSONDecodeError, ValueError):
            pred = {}

        try:
            gold = json.loads(gold_str)
        except (json.JSONDecodeError, ValueError):
            gold = {}

        for key in PII_CATEGORIES:
            pred_vals = pred.get(key, [])
            gold_vals = gold.get(key, [])
            pred_set = set(pred_vals) if isinstance(pred_vals, list) else set()
            gold_set = set(gold_vals) if isinstance(gold_vals, list) else set()
            cat_stats[key]["tp"] += len(pred_set & gold_set)
            cat_stats[key]["fp"] += len(pred_set - gold_set)
            cat_stats[key]["fn"] += len(gold_set - pred_set)

    total_tp = total_fp = total_fn = 0
    per_category: dict[str, CategoryResult] = {}
    for key in PII_CATEGORIES:
        tp = cat_stats[key]["tp"]
        fp = cat_stats[key]["fp"]
        fn = cat_stats[key]["fn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_category[key] = CategoryResult(precision=p, recall=r, f1=f1, tp=tp, fp=fp, fn=fn)

    mp = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    mr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    mf1 = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0

    return EvalResult(
        n_samples=actual,
        json_parse_rate=json_ok / actual,
        micro_precision=mp,
        micro_recall=mr,
        micro_f1=mf1,
        per_category=per_category,
    )


def log_to_wandb(result: EvalResult, run: Run, step: int | None = None) -> None:
    """Log EvalResult to W&B: scalars + a per-category Table sorted by F1 (weakest first)."""
    import wandb

    log_dict: dict[str, object] = {
        "eval/json_parse_rate": result.json_parse_rate,
        "eval/micro_precision": result.micro_precision,
        "eval/micro_recall": result.micro_recall,
        "eval/micro_f1": result.micro_f1,
    }
    for key, m in result.per_category.items():
        log_dict[f"eval/{key}/f1"] = m.f1
        log_dict[f"eval/{key}/precision"] = m.precision
        log_dict[f"eval/{key}/recall"] = m.recall

    rows = sorted(
        [
            [key, m.f1, m.precision, m.recall, m.tp, m.fp, m.fn]
            for key, m in result.per_category.items()
        ],
        key=lambda r: r[1],
    )
    table = wandb.Table(
        columns=["category", "f1", "precision", "recall", "tp", "fp", "fn"],
        data=rows,
    )
    log_dict["eval/per_category"] = table
    log_dict["eval/per_category_f1_chart"] = wandb.plot.bar(
        table, "category", "f1", title="Per-category F1 (weakest first)"
    )

    if step is not None:
        run.log(log_dict, step=step)
    else:
        run.log(log_dict)


def print_report(result: EvalResult) -> None:
    """Print a per-category breakdown to stdout, sorted by F1 ascending (weakest first)."""
    sorted_cats = sorted(result.per_category.items(), key=lambda kv: kv[1].f1)
    header = f"{'category':<22} {'F1':>6} {'Prec':>6} {'Rec':>6} {'TP':>5} {'FP':>5} {'FN':>5}"
    sep = "-" * len(header)
    print(f"\n[fox-co] Eval results ({result.n_samples} samples)  json_parse_rate={result.json_parse_rate:.3f}")
    print(sep)
    print(header)
    print(sep)
    for key, m in sorted_cats:
        print(
            f"{key:<22} {m.f1:>6.3f} {m.precision:>6.3f} {m.recall:>6.3f}"
            f" {m.tp:>5} {m.fp:>5} {m.fn:>5}"
        )
    print(sep)
    print(
        f"{'micro avg':<22} {result.micro_f1:>6.3f}"
        f" {result.micro_precision:>6.3f} {result.micro_recall:>6.3f}"
    )
    print()


# ---------------------------------------------------------------------------


def _parse_dataset_mapper(raw: str | None) -> dict[str, str] | None:
    """Parse the DATASET_MAPPER env var into a {role: column} dict."""
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
    """Map a dataset row to the chat-format {"messages": [...]} SFTTrainer expects."""
    if mapper:
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
    if "messages" in example and isinstance(example["messages"], list):
        return {"messages": example["messages"]}
    if "conversations" in example and isinstance(example["conversations"], list):
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        return {
            "messages": [
                {"role": role_map.get(turn["from"], turn["from"]), "content": turn["value"]}
                for turn in example["conversations"]
            ]
        }
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
    dataset_name: str,
    dataset_slice: int,
    mapper: dict[str, str] | None,
    eval_split_ratio: float = 0.1,
) -> tuple[Dataset, Dataset | None]:
    """Load DATASET from HF Hub, map to chat format, and resolve an eval split.

    Eval split priority:
      1. HF dataset の "validation" split（存在すれば）
      2. HF dataset の "test" split（存在すれば）
      3. train から eval_split_ratio で切り出し（fallback）
      4. eval_split_ratio=0 なら eval なし
    """
    from datasets import get_dataset_split_names, load_dataset

    split_spec = f"train[:{dataset_slice}]" if dataset_slice > 0 else "train"
    train_raw = load_dataset(dataset_name, split=split_spec)

    # 1-2. HF dataset に既存の eval split があるか確認
    eval_raw: Dataset | None = None
    try:
        available = get_dataset_split_names(dataset_name)
    except Exception:
        available = []

    for eval_split in ("validation", "test"):
        if eval_split in available:
            eval_raw = load_dataset(dataset_name, split=eval_split)
            print(f"[fox-co] Using existing '{eval_split}' split as eval ({len(eval_raw)} rows)")
            break

    # 3. fallback: train から切り出し
    if eval_raw is None and eval_split_ratio > 0:
        splits = train_raw.train_test_split(test_size=eval_split_ratio, seed=42)
        train_raw = splits["train"]
        eval_raw = splits["test"]
        print(f"[fox-co] No eval split found; carved out {len(eval_raw)} rows from train")

    train_ds = train_raw.map(
        lambda ex: _to_messages(ex, mapper), remove_columns=train_raw.column_names
    )
    if eval_raw is not None:
        eval_ds: Dataset | None = eval_raw.map(
            lambda ex: _to_messages(ex, mapper), remove_columns=eval_raw.column_names
        )
    else:
        eval_ds = None

    print(f"[fox-co] train={len(train_ds)} eval={len(eval_ds) if eval_ds else 0}")
    return train_ds, eval_ds


def load_base_model(model_id: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load tokenizer + model with the fox-co chat template injected."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Inject the fox-co PII extraction chat template.
    # NOTE: if submitting to HF Jobs, inline this template as a string constant
    # since HF Jobs uploads a single script file.
    template_path = pathlib.Path(__file__).parent / "chat_template.jinja"
    tok.chat_template = template_path.read_text(encoding="utf-8")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
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
    """Canonical LFM2 LoRA recipe per https://docs.liquid.ai/lfm/fine-tuning/unsloth."""
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
        gradient_accumulation_steps=4,
        max_steps=max_steps,
        learning_rate=lr,
        warmup_steps=max(10, max_steps // 20),
        lr_scheduler_type="linear",
        optim="adamw_torch",
        weight_decay=0.01,
        logging_steps=max(1, max_steps // 50),
        report_to=["wandb"],
        bf16=True,
        save_strategy="steps",
        save_steps=max(50, max_steps // 10),
        save_total_limit=3,
        max_length=2048,
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
    """Save LoRA adapter as backup, then merge into base and save standalone model."""
    adapter_dir = f"{output_dir}-adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)

    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    print(f"[fox-co] Merged checkpoint saved to {output_dir}")


def push_checkpoint(folder: str, repo_id: str, run: Run) -> None:
    """Upload merged checkpoint to Hub and stamp commit SHA in W&B summary."""
    from huggingface_hub import HfApi

    print(f"[fox-co] Pushing to https://huggingface.co/{repo_id} ...")
    api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True)
    commit = api.upload_folder(
        folder_path=folder,
        repo_id=repo_id,
        commit_message="fox-co trained merged checkpoint",
    )
    run.summary["hf_revision"] = commit.oid
    run.summary["hf_repo"] = repo_id
    print(f"[fox-co] Push complete. Hub revision: {commit.oid}")


def main() -> None:
    import torch
    import wandb

    dataset_name = os.environ.get("DATASET")
    if not dataset_name:
        raise SystemExit(
            "[fox-co] DATASET env var is required. "
            "Set it to an HF Hub dataset id, e.g. DATASET=your-org/fox-co-pii"
        )

    model_id = os.environ.get("MODEL_ID", "LiquidAI/LFM2-350M")
    dataset_slice = int(os.environ.get("DATASET_SLICE", "1024"))
    eval_split_ratio = float(os.environ.get("EVAL_SPLIT_RATIO", "0.1"))
    eval_samples = int(os.environ.get("EVAL_SAMPLES", "50"))
    skip_training = os.environ.get("SKIP_TRAINING", "0").strip() == "1"
    max_steps = int(os.environ.get("MAX_STEPS", "200"))
    batch_size = int(os.environ.get("BATCH_SIZE", "4"))
    lr = float(os.environ.get("LR", "2e-4"))
    output_dir = os.environ.get("OUTPUT_DIR", "/tmp/lfm2-fox-co")
    push_to_hub = os.environ.get("PUSH_TO_HUB")
    mapper = _parse_dataset_mapper(os.environ.get("DATASET_MAPPER"))

    print(
        f"[fox-co] torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"model={model_id} dataset={dataset_name} slice={dataset_slice} "
        f"skip_training={skip_training} max_steps={max_steps} bs={batch_size} lr={lr}"
    )

    train_ds, eval_ds = load_training_dataset(
        dataset_name, dataset_slice, mapper, eval_split_ratio
    )

    tags = ["text", "lfm2", "lora", "fox-co", "pii"]
    if skip_training:
        tags.append("eval-only")
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "hack-the-liquid-way"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=os.environ.get("WANDB_RUN_NAME"),
        tags=tags,
        config={
            "model": model_id,
            "dataset": dataset_name,
            "dataset_slice": dataset_slice,
            "dataset_mapper": mapper,
            "eval_split_ratio": eval_split_ratio,
            "eval_samples": eval_samples,
            "skip_training": skip_training,
            "max_steps": max_steps if not skip_training else 0,
            "batch_size": batch_size,
            "lr": lr,
        },
    )
    print(f"[fox-co] W&B run URL: {run.url}")

    model, tok = load_base_model(model_id)

    if not skip_training:
        trainer = build_trainer(
            model, tok, train_ds, max_steps=max_steps, batch_size=batch_size, lr=lr, output_dir=output_dir
        )
        print("[fox-co] Starting training...")
        out = trainer.train()
        print(f"[fox-co] FINAL TRAIN LOSS: {out.training_loss}")
        eval_target = trainer.model
    else:
        print("[fox-co] SKIP_TRAINING=1: skipping training, running eval only.")
        eval_target = model

    if eval_ds is not None and eval_samples > 0:
        result = run_eval(eval_target, tok, eval_ds, eval_samples)
        print_report(result)
        log_to_wandb(result, run)

    if not skip_training:
        save_merged_checkpoint(trainer, tok, output_dir)
        if push_to_hub:
            push_checkpoint(output_dir, push_to_hub, run)

    wandb.finish()
    print("[fox-co] DONE")


if __name__ == "__main__":
    main()
