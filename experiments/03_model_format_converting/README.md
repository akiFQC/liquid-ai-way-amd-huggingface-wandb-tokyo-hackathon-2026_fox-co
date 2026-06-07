# 03 モデルフォーマット変換 (GGUF)

Fine-tune済みモデルを llama.cpp 互換の GGUF 形式に変換し、HF Hub へアップロードするスクリプト群です。

## ファイル構成

| ファイル | 役割 |
|---|---|
| `convert_to_gguf.py` | HF モデル → GGUF 変換・アップロード（UV script） |
| `launch_hf_job.sh` | HF Jobs へ submit するラッパースクリプト |

---

## 変換フロー

```
SOURCE_MODEL (HF Hub)
  ↓ snapshot_download
hf_model/                         ← safetensors + tokenizer
  ↓ convert_hf_to_gguf.py (llama.cpp)
model-F16.gguf                    ← 中間 F16 GGUF
  ↓ llama-quantize × 3
model-Q4_K_M.gguf (Int4)
model-Q8_0.gguf   (Int8)
model-BF16.gguf   (BFloat16)
  ↓ api.upload_file()
TARGET_REPO (HF Hub)              ← 全4ファイル + README.md
```

---

## 実行方法

### HF Jobs（推奨）

`.env` に `HF_TOKEN`（**write スコープ必須**）を設定した上で:

```bash
# デフォルト設定で実行
./experiments/03_model_format_converting/launch_hf_job.sh

# 変数を明示指定
SOURCE_MODEL=akiFQC/LFM2.5-1.2B-JP-202606-Conf-Extract \
TARGET_REPO=akiFQC/LFM2.5-1.2B-JP-202606-Conf-Extract-GGUF \
QUANT_TYPES=Q4_K_M,Q8_0,BF16 \
  ./experiments/03_model_format_converting/launch_hf_job.sh

# dry run（submit せずにゲートと引数を確認）
DRY_RUN=1 \
  ./experiments/03_model_format_converting/launch_hf_job.sh
```

スクリプトが自動で行うこと:

- `.env` の読み込みと `HF_TOKEN` チェック
- `TARGET_REPO` への書き込み権限事前確認（`PUSH_TO_HUB` 相当）
- secrets を一時ファイル経由で渡す（argv / `ps aux` に露出させない）

### ローカル実行

cmake と git が入った Linux/Mac 環境なら `uv run` で直接実行できます:

```bash
HF_TOKEN=$HF_TOKEN \
SOURCE_MODEL=akiFQC/LFM2.5-1.2B-JP-202606-Conf-Extract \
TARGET_REPO=akiFQC/LFM2.5-1.2B-JP-202606-Conf-Extract-GGUF \
  uv run experiments/03_model_format_converting/convert_to_gguf.py
```

---

## 環境変数一覧

### `convert_to_gguf.py`

| 変数 | デフォルト | 説明 |
|---|---|---|
| `HF_TOKEN` | **必須** | write スコープの HF Token |
| `SOURCE_MODEL` | `akiFQC/LFM2.5-1.2B-JP-202606-Conf-Extract` | 変換元 HF モデル repo id |
| `TARGET_REPO` | `akiFQC/LFM2.5-1.2B-JP-202606-Conf-Extract-GGUF` | アップロード先 HF repo id |
| `QUANT_TYPES` | `Q4_K_M,Q8_0,BF16` | カンマ区切り量子化タイプ一覧 |
| `CREATE_PUSH_TARGET` | `1` | `1` = ターゲット repo を自動作成 |
| `WORK_DIR` | `/tmp/gguf-convert` | 一時ファイルの保存先 |

**量子化タイプ対応表:**

| `QUANT_TYPES` 値 | llama-quantize タイプ | 用途 |
|---|---|---|
| `Q4_K_M` | Q4_K_M | 4-bit 量子化（Int4）— 最小サイズ、on-device 推論向け |
| `Q8_0` | Q8_0 | 8-bit 量子化（Int8）— バランス型 |
| `BF16` | BF16 | BFloat16（量子化なし、精度優先） |

### `launch_hf_job.sh`

| 変数 | デフォルト | 説明 |
|---|---|---|
| `HF_FLAVOR` | `l4x1` | GPU 種別（[料金表](https://huggingface.co/docs/hub/jobs-pricing)）|
| `HF_TIMEOUT` | `1h` | タイムアウト（この時点で課金停止） |
| `HF_NAMESPACE` | — | 組織名前空間（省略時は個人アカウント） |
| `DRY_RUN` | — | `1` にすると submit をスキップしてコマンドを表示 |

> **Note:** WANDB は不要です。変換ジョブは学習メトリクスを記録しません。

---

## アップロードされるファイル

変換完了後、`TARGET_REPO` に以下が作成されます:

| ファイル | サイズ目安 | 説明 |
|---|---|---|
| `*-F16.gguf` | ~2.4 GB | 中間 F16。他の量子化形式への再変換に使用 |
| `*-Q4_K_M.gguf` | ~0.7 GB | Int4 量子化。on-device（llama.cpp）向け |
| `*-Q8_0.gguf` | ~1.3 GB | Int8 量子化。精度と速度のバランス型 |
| `*-BF16.gguf` | ~2.4 GB | BFloat16。高精度推論向け |
| `README.md` | — | 自動生成モデルカード |

---

## 注意事項

- `HF_TOKEN` は **write スコープ** が必要です（`TARGET_REPO` へのアップロードのため）
- cmake と git がインストールされた環境が必要です（HF Jobs コンテナには標準で含まれます）
- llama.cpp の cmake ビルドに数分かかるため、`HF_TIMEOUT` には余裕を持たせてください
- `QUANT_TYPES` を絞ることで変換時間を短縮できます（例: `QUANT_TYPES=Q8_0`）
