#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets"]
# ///
"""
Step 5: SFT用データ整形

mixed_dataset（および元の3データセット）を train.py が受け付ける
messages 列フォーマットに変換し、train/validation split で保存する。
HF Upload（Step 6）向けに metadata.json も生成する。

出力先: experiments/data/sft_dataset/
  train/          HuggingFace Arrow
  validation/     HuggingFace Arrow
  metadata.json   HF Upload 用メタ情報
"""

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk

# chat_template.jinja の default system prompt と同じ内容
SYSTEM_PROMPT = (
    "あなたはテキストから社外秘の固有表現を抽出するアシスタントです。"
    "入力テキストを分析し、以下の11カテゴリの機密情報を抽出して、必ずJSON形式のみで出力してください。\n\n"
    "カテゴリ定義:\n"
    "- address: 住所・所在地\n"
    "- company_name: 企業・研究機関・組織名\n"
    "- email_address: メールアドレス\n"
    "- human_name: 人名\n"
    "- phone_number: 電話番号\n"
    "- account_identifier: アカウント識別子（ユーザーID・アカウント名・従業員番号・社会保障番号・マイナンバー等）\n"
    "- network_identifier: ネットワーク識別情報（IPアドレス・MACアドレス・内部ドメイン・ホスト名）\n"
    "- system_config: システム構成情報（ファイルパス・ディレクトリ構造・DBテーブル/カラム名）\n"
    "- project_info: プロジェクト関連情報（プロジェクト名・開発コードネーム・未発表の製品/機能名）\n"
    "- financial_info: 金額・財務情報（売上・原価・利益率・契約金額・個人の給与/報酬額）\n"
    "- transaction_id: 取引管理番号（契約書番号・請求書番号・見積書番号・顧客管理ID）\n\n"
    "出力形式（全キーを必ず含め、該当なしは空リスト）:\n"
    '{"address": [], "company_name": [], "email_address": [], "human_name": [], '
    '"phone_number": [], "account_identifier": [], "network_identifier": [], '
    '"system_config": [], "project_info": [], "financial_info": [], "transaction_id": []}'
)

ALL_KEYS = [
    "address", "company_name", "email_address", "human_name", "phone_number",
    "account_identifier", "network_identifier", "system_config", "project_info",
    "financial_info", "transaction_id",
]

DATA_ROOT = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_ROOT / "sft_dataset"

ORIGINAL_SOURCES = {
    "OpenPII 1.5M (ja)": DATA_ROOT / "openpii_processed",
    "ner-wikipedia-dataset": DATA_ROOT / "ner_wikipedia_processed",
    "synthetic": DATA_ROOT / "synthetic_processed",
}
MIXED_SOURCE = DATA_ROOT / "mixed_dataset"


def row_to_messages(row: dict) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": row["input_text"]},
            {"role": "assistant", "content": row["annotation_json"]},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT用データ整形スクリプト")
    parser.add_argument("--no-include-originals", action="store_true",
                        help="元の3データセットを含めず mixed_dataset のみ使用する")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="validation 割合 (デフォルト: 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                        help="乱数シード (デフォルト: 42)")
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val-ratio は 0.0 より大きく 1.0 未満で指定してください")

    rng = random.Random(args.seed)

    # --- データセットのロード ---
    datasets_to_merge = []
    source_names_used = []

    # mixed_dataset は必須
    if not MIXED_SOURCE.exists():
        print(f"エラー: mixed_dataset が見つかりません: {MIXED_SOURCE}")
        print("先に augment_datasets.py を実行してください。")
        raise SystemExit(1)
    ds_mixed = load_from_disk(str(MIXED_SOURCE))
    datasets_to_merge.append(ds_mixed)
    source_names_used.append("mixed")
    print(f"[LOAD] mixed_dataset: {len(ds_mixed):,} rows")

    # 元の3データセット（オプション）
    if not args.no_include_originals:
        for name, path in ORIGINAL_SOURCES.items():
            if not path.exists():
                print(f"[SKIP] {name}: {path} が見つかりません")
                continue
            ds = load_from_disk(str(path))
            datasets_to_merge.append(ds)
            source_names_used.append(name)
            print(f"[LOAD] {name}: {len(ds):,} rows")

    # --- 結合・messages 変換 ---
    combined = concatenate_datasets(datasets_to_merge)
    print(f"\n合計 {len(combined):,} 行を messages 形式に変換中...")

    ds_sft = combined.map(
        row_to_messages,
        remove_columns=combined.column_names,
        desc="変換",
    )

    # --- シャッフル ---
    indices = list(range(len(ds_sft)))
    rng.shuffle(indices)
    ds_sft = ds_sft.select(indices)

    # --- train / validation split ---
    n_total = len(ds_sft)
    n_val = max(1, round(n_total * args.val_ratio))
    n_train = n_total - n_val

    ds_train = ds_sft.select(range(n_train))
    ds_val   = ds_sft.select(range(n_train, n_total))

    print(f"  train: {n_train:,} rows")
    print(f"  validation: {n_val:,} rows")

    # --- 保存 ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "train"
    val_path   = OUTPUT_DIR / "validation"

    ds_train.save_to_disk(str(train_path))
    ds_val.save_to_disk(str(val_path))

    # --- metadata.json ---
    metadata = {
        "dataset_name": "japanese-confidential-information-extraction-sft",
        "description": (
            "SFT dataset for Japanese confidential information extraction. "
            "Each example is a [system, user, assistant] chat message list. "
            "The user message is Japanese text; the assistant message is a JSON string "
            "with 11 PII/sensitive-info categories."
        ),
        "language": "ja",
        "task": "named-entity-recognition",
        "license": "see source datasets",
        "categories": ALL_KEYS,
        "source_datasets": source_names_used,
        "num_rows": {
            "train": n_train,
            "validation": n_val,
            "total": n_total,
        },
        "format": {
            "column": "messages",
            "roles": ["system", "user", "assistant"],
            "system_prompt": "see chat_template.jinja",
        },
        "hf_upload_notes": (
            "Push to Hub with: "
            "DatasetDict({'train': ds_train, 'validation': ds_val}).push_to_hub('<repo_id>')"
        ),
    }
    meta_path = OUTPUT_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    print(f"\n保存完了: {OUTPUT_DIR}")
    print(f"  {train_path.name}/  ({n_train:,} rows)")
    print(f"  {val_path.name}/    ({n_val:,} rows)")
    print(f"  metadata.json")


if __name__ == "__main__":
    main()
