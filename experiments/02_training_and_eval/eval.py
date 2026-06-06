"""PII extraction evaluation utilities for fox-co.

Import-safe: no torch / transformers at module level.
Heavy imports are deferred inside run_eval() so this module can be imported
from notebooks, scripts, or preflight checks without pulling GPU dependencies.

Typical usage::

    from eval import run_eval, log_to_wandb, print_report
    result = run_eval(model, tok, eval_ds, n_samples=100)
    print_report(result)
    log_to_wandb(result, wandb_run)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    from wandb.sdk.wandb_run import Run

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

    # Per-category table sorted by F1 ascending (weakest first).
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
