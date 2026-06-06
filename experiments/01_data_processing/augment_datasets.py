#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "tabulate", "tqdm"]
# ///
"""
Step 4: データの混合とデータ拡張

3つのprocessed datasetsを組み合わせて augmented dataset を生成する。
アルゴリズム:
  1. データソースごとにキューを作り、重複なしサンプリング（weighted shuffle）
  2. 2〜4件をランダムに選び、テキストをランダム順で結合（区切り: 。 or \\n）
  3. 選ばれたサンプルのラベルを union して annotation_json を構成
  4. ラベル統計を表示し、japanese mixed confidential information extraction dataset として保存

ラベルバランス改善:
  IDF重み付きサンプリングにより、レアカテゴリ（network_identifier, system_config,
  transaction_id 等）を含むサンプルが優先的に選ばれるよう調整する。
"""

import argparse
import json
import math
import random
from collections import defaultdict, deque
from pathlib import Path

from datasets import Dataset, load_from_disk
from tabulate import tabulate
from tqdm import tqdm

ALL_KEYS = [
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
SOURCES = {
    "OpenPII 1.5M (ja)": DATA_ROOT / "openpii_processed",
    "ner-wikipedia-dataset": DATA_ROOT / "ner_wikipedia_processed",
    "synthetic": DATA_ROOT / "synthetic_processed",
}
OUTPUT_DIR = DATA_ROOT / "mixed_dataset"


def _empty_annotation() -> dict:
    return {k: [] for k in ALL_KEYS}


def compute_category_counts(all_annotations: list[dict]) -> dict[str, int]:
    """全サンプルのカテゴリ別非空行数を集計する。"""
    counts: dict[str, int] = defaultdict(int)
    for ann in all_annotations:
        for cat, vals in ann.items():
            if vals:
                counts[cat] += 1
    return dict(counts)


def compute_sample_weight(ann: dict, cat_counts: dict[str, int], total: int) -> float:
    """IDF重み: レアカテゴリを含むサンプルほど高い重みを付ける。"""
    weight = 0.0
    for cat, vals in ann.items():
        n = cat_counts.get(cat, 0)
        if vals and n > 0:
            weight += math.log(total / n)
    return max(weight, 1e-9)


def weighted_shuffle(indices: list[int], weights: list[float], rng: random.Random) -> list[int]:
    """Efraimidis-Spirakis weighted shuffle (sampling without replacement)。
    key = -log(U) / w が小さい順 → 重みが大きいほど先頭に来やすい。
    """
    keys = [-math.log(rng.random()) / w for w in weights]
    return sorted(indices, key=lambda i: keys[i])


def merge_annotations(ann_list: list[dict]) -> dict:
    """複数の annotation_json を重複なしで union する。"""
    merged = _empty_annotation()
    for ann in ann_list:
        for key in ALL_KEYS:
            for val in ann.get(key, []):
                if val not in merged[key]:
                    merged[key].append(val)
    return merged


def generate_augmented_samples(
    rows_by_source: dict[str, list[dict]],
    n_augmented: int,
    rng: random.Random,
) -> list[dict]:
    """2〜4件結合の augmented サンプルを n_augmented 件生成する。"""

    # 1. 全ソースの annotation を集めてカテゴリ頻度を計算
    all_annotations = []
    for rows in rows_by_source.values():
        for row in rows:
            all_annotations.append(json.loads(row["annotation_json"]))
    total_rows = len(all_annotations)
    cat_counts = compute_category_counts(all_annotations)

    # 2. ソースごとのサンプル重みを計算
    weights_by_source: dict[str, list[float]] = {}
    for name, rows in rows_by_source.items():
        weights_by_source[name] = [
            compute_sample_weight(json.loads(row["annotation_json"]), cat_counts, total_rows)
            for row in rows
        ]

    # 3. ソースごとに weighted_shuffle 済みの deque を作成
    queues: dict[str, deque] = {}
    for name, rows in rows_by_source.items():
        indices = list(range(len(rows)))
        ordered = weighted_shuffle(indices, weights_by_source[name], rng)
        queues[name] = deque(ordered)

    def refill_queue(name: str) -> None:
        """キューが空になったら再 weighted_shuffle して補充する。"""
        indices = list(range(len(rows_by_source[name])))
        ordered = weighted_shuffle(indices, weights_by_source[name], rng)
        queues[name].extend(ordered)

    # ソース選択の重みはデータサイズ比（大きいソースが多く選ばれる）
    source_names = list(rows_by_source.keys())
    source_sizes = [len(rows_by_source[name]) for name in source_names]

    augmented: list[dict] = []
    for _ in tqdm(range(n_augmented), desc="augmenting"):
        k = rng.randint(2, 4)
        chosen_sources = rng.choices(source_names, weights=source_sizes, k=k)

        selected_rows: list[dict] = []
        for src in chosen_sources:
            if not queues[src]:
                refill_queue(src)
            idx = queues[src].popleft()
            selected_rows.append(rows_by_source[src][idx])

        # テキストをランダムに並び替えて結合
        rng.shuffle(selected_rows)
        delimiter = rng.choice(["。", "\n"])
        combined_text = delimiter.join(row["input_text"] for row in selected_rows)

        # annotation_json を union
        ann_list = [json.loads(row["annotation_json"]) for row in selected_rows]
        merged = merge_annotations(ann_list)

        augmented.append(
            {
                "input_text": combined_text,
                "annotation_json": json.dumps(merged, ensure_ascii=False),
            }
        )

    return augmented


def collect_negative_samples(
    rows_by_source: dict[str, list[dict]],
    n: int,
    rng: random.Random,
) -> list[dict]:
    """全カテゴリが空の行をまとめて n 件サンプリングする。
    n が利用可能数を超える場合はリピートサンプリングで補う。
    """
    pool = [
        row
        for rows in rows_by_source.values()
        for row in rows
        if not any(json.loads(row["annotation_json"]).values())
    ]
    if not pool:
        return []
    if n <= len(pool):
        return rng.sample(pool, n)
    # 利用可能数が足りない場合はリピートサンプリング
    return rng.choices(pool, k=n)


# --- 統計表示 (stats_datasets.py と同スタイル) ---

def compute_stats(dataset) -> dict:
    total_rows = len(dataset)
    entity_count: dict[str, int] = defaultdict(int)
    row_count: dict[str, int] = defaultdict(int)

    for row in dataset:
        ann = json.loads(row["annotation_json"])
        for cat in ALL_KEYS:
            vals = ann.get(cat, [])
            if vals:
                entity_count[cat] += len(vals)
                row_count[cat] += 1

    return {"total_rows": total_rows, "entity_count": entity_count, "row_count": row_count}


def print_stats(name: str, stats: dict) -> None:
    total_rows = stats["total_rows"]
    rows = []
    for cat in ALL_KEYS:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="データ混合・拡張スクリプト")
    parser.add_argument("--n-augmented", type=int, default=10_000, help="生成するサンプル数 (デフォルト: 10000)")
    parser.add_argument("--negative-ratio", type=float, default=0.1,
                        help="ネガティブサンプル(全カテゴリ空)の割合 0.0〜1.0 (デフォルト: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード (デフォルト: 42)")
    args = parser.parse_args()

    if not 0.0 <= args.negative_ratio < 1.0:
        parser.error("--negative-ratio は 0.0 以上 1.0 未満で指定してください")

    rng = random.Random(args.seed)

    # データセットをロード
    rows_by_source: dict[str, list[dict]] = {}
    for name, path in SOURCES.items():
        if not path.exists():
            print(f"[SKIP] {name}: {path} が見つかりません")
            continue
        ds = load_from_disk(str(path))
        rows_by_source[name] = list(ds)
        print(f"[LOAD] {name}: {len(ds):,} rows")

    if not rows_by_source:
        print("エラー: ロードできるデータセットがありません。prepare_datasets.py を先に実行してください。")
        raise SystemExit(1)

    # 件数の分割（ネガティブ比率を考慮）
    n_negative = round(args.n_augmented * args.negative_ratio)
    n_positive = args.n_augmented - n_negative

    # ポジティブ拡張サンプルを生成
    print(f"\nポジティブサンプル {n_positive:,} 件を生成中 (seed={args.seed}) ...")
    augmented = generate_augmented_samples(rows_by_source, n_positive, rng)

    # ネガティブサンプルを収集
    if n_negative > 0:
        negatives = collect_negative_samples(rows_by_source, n_negative, rng)
        if negatives:
            print(f"ネガティブサンプル {len(negatives):,} 件を追加 (全カテゴリ空)")
            augmented.extend(negatives)
        else:
            print("[WARN] ネガティブサンプルが見つかりませんでした")

    # シャッフルして順序バイアスを除去
    rng.shuffle(augmented)

    # HuggingFace Dataset に変換
    ds_augmented = Dataset.from_list(augmented)

    # 統計表示
    stats = compute_stats(ds_augmented)
    print_stats("japanese mixed confidential information extraction dataset (augmented)", stats)

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ds_augmented.save_to_disk(str(OUTPUT_DIR))
    print(f"\n保存完了: {OUTPUT_DIR}  ({len(ds_augmented):,} rows)")


if __name__ == "__main__":
    main()
