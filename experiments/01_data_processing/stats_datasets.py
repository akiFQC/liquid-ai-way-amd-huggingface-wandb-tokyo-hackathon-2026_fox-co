#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "tabulate"]
# ///
"""3データセットのラベル分布統計を表示する。"""

import json
from collections import defaultdict
from pathlib import Path

from datasets import load_from_disk
from tabulate import tabulate

CATEGORIES = [
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

DATA_ROOT = Path(__file__).parent.parent / "data"
DATASETS = {
    "OpenPII 1.5M (ja)": DATA_ROOT / "openpii_processed",
    "ner-wikipedia-dataset": DATA_ROOT / "ner_wikipedia_processed",
    "synthetic": DATA_ROOT / "synthetic_processed",
}


def compute_stats(dataset) -> dict:
    total_rows = len(dataset)
    entity_count: dict[str, int] = defaultdict(int)
    row_count: dict[str, int] = defaultdict(int)

    for row in dataset:
        ann = json.loads(row["annotation_json"])
        for cat in CATEGORIES:
            vals = ann.get(cat, [])
            if vals:
                entity_count[cat] += len(vals)
                row_count[cat] += 1

    return {"total_rows": total_rows, "entity_count": entity_count, "row_count": row_count}


def print_stats(name: str, stats: dict) -> None:
    total_rows = stats["total_rows"]
    rows = []
    for cat in CATEGORIES:
        ec = stats["entity_count"][cat]
        rc = stats["row_count"][cat]
        cov = rc / total_rows * 100 if total_rows > 0 else 0.0
        rows.append([cat, f"{ec:,}", f"{rc:,}", f"{cov:.1f}%"])

    print(f"\n[{name}] {total_rows:,} rows")
    print(
        tabulate(
            rows,
            headers=["カテゴリ", "件数", "行数", "行カバー率"],
            tablefmt="simple",
            colalign=("left", "right", "right", "right"),
        )
    )


def print_combined(all_stats: dict[str, dict]) -> None:
    names = list(all_stats.keys())
    total_rows = sum(s["total_rows"] for s in all_stats.values())
    rows = []
    for cat in CATEGORIES:
        ec_total = sum(s["entity_count"][cat] for s in all_stats.values())
        rc_total = sum(s["row_count"][cat] for s in all_stats.values())
        cov = rc_total / total_rows * 100 if total_rows > 0 else 0.0
        rows.append([cat, f"{ec_total:,}", f"{rc_total:,}", f"{cov:.1f}%"])

    print(f"\n[合計 ({' + '.join(names)})] {total_rows:,} rows")
    print(
        tabulate(
            rows,
            headers=["カテゴリ", "件数合計", "行数合計", "行カバー率"],
            tablefmt="simple",
            colalign=("left", "right", "right", "right"),
        )
    )


def main() -> None:
    all_stats: dict[str, dict] = {}
    for name, path in DATASETS.items():
        if not path.exists():
            print(f"[SKIP] {name}: {path} が見つかりません")
            continue
        ds = load_from_disk(str(path))
        stats = compute_stats(ds)
        print_stats(name, stats)
        all_stats[name] = stats

    if len(all_stats) > 1:
        print_combined(all_stats)


if __name__ == "__main__":
    main()
