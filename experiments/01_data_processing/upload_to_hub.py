#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "huggingface_hub"]
# ///
"""
Step 6: HuggingFace Hub へのアップロード

experiments/data/sft_dataset/ を HF Hub の dataset リポジトリに push する。

事前準備:
  - HF_TOKEN 環境変数に WRITE スコープのトークンを設定する
    export HF_TOKEN=hf_...
  - または .env ファイルに HF_TOKEN=hf_... を記述する

使用例:
  uv run experiments/01_data_processing/upload_to_hub.py \\
      --repo-id <your-hf-username>/japanese-confidential-information-extraction-sft
"""

import argparse
import json
import os
import sys
from pathlib import Path

DATA_ROOT = Path(__file__).parent.parent / "data"
SFT_DIR = DATA_ROOT / "sft_dataset"


def _load_dotenv(path: Path) -> None:
    """シンプルな .env ローダー（python-dotenv 不要）。"""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


def main() -> None:
    _load_dotenv(Path(__file__).parent.parent.parent / ".env")

    parser = argparse.ArgumentParser(description="SFT データセットを HF Hub へアップロード")
    parser.add_argument(
        "--repo-id", required=True,
        help="HF Hub のリポジトリ名 (例: your-username/japanese-pii-extraction-sft)",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="プライベートリポジトリとして作成する",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="実際にはアップロードせず、内容の確認のみ行う",
    )
    args = parser.parse_args()

    # HF_TOKEN チェック
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("エラー: HF_TOKEN が設定されていません。")
        print("  export HF_TOKEN=hf_...  または .env に記述してください。")
        print("  トークンは https://huggingface.co/settings/tokens で発行（WRITE スコープ必須）。")
        sys.exit(1)

    # sft_dataset の存在確認
    if not SFT_DIR.exists():
        print(f"エラー: {SFT_DIR} が見つかりません。")
        print("先に format_sft.py を実行してください。")
        sys.exit(1)

    # metadata 読み込み
    meta_path = SFT_DIR / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    print(f"アップロード先: https://huggingface.co/datasets/{args.repo_id}")
    print(f"プライベート: {args.private}")
    if metadata:
        nr = metadata.get("num_rows", {})
        print(f"データ数: train={nr.get('train', '?'):,}  validation={nr.get('validation', '?'):,}")

    if args.dry_run:
        print("\n[DRY RUN] アップロードはスキップされました。")
        return

    # アップロード実行
    from datasets import DatasetDict, load_from_disk

    print("\nデータセットをロード中...")
    ds_dict = DatasetDict({
        "train":      load_from_disk(str(SFT_DIR / "train")),
        "validation": load_from_disk(str(SFT_DIR / "validation")),
    })

    print(f"push_to_hub: {args.repo_id} ...")
    ds_dict.push_to_hub(
        args.repo_id,
        private=args.private,
        token=token,
    )

    print(f"\nアップロード完了: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
