# 02 Training & Eval

LFM2 の LoRA fine-tune と PII 抽出タスクの評価を行うスクリプト群です。

## ファイル構成

| ファイル | 役割 |
|---|---|
| `train.py` | 学習・評価のエントリーポイント（UV script） |
| `eval.py` | 評価ロジック（importable モジュール） |
| `chat_template.jinja` | fox-co PII 抽出タスク用の chat template |

---

## train.py — 学習 + 評価

### 最小実行例

```bash
DATASET= \
  uv run experiments/02_training_and_eval/train.py
```

### 評価のみ（学習スキップ）

ベースモデルや公開済み fine-tune の評価だけ行いたい場合:

```bash
DATASET=your-org/japanese-pii-sft \
SKIP_TRAINING=1 \
  uv run experiments/02_training_and_eval/train.py

# fine-tune 済みチェックポイントを評価
MODEL_ID=your-org/fox-co-finetuned \
DATASET=your-org/japanese-pii-sft \
SKIP_TRAINING=1 \
  uv run experiments/02_training_and_eval/train.py
```

### 主な環境変数

| 変数 | デフォルト | 説明 |
|---|---|---|
| `DATASET` | **必須** | HF Hub dataset id |
| `MODEL_ID` | `LiquidAI/LFM2-350M` | ベースモデル |
| `DATASET_SLICE` | `1024` | 学習に使う最大行数（`0` = 全件） |
| `MAX_STEPS` | `200` | 学習ステップ数 |
| `BATCH_SIZE` | `4` | per-device バッチサイズ |
| `LR` | `2e-4` | 学習率 |
| `SKIP_TRAINING` | `0` | `1` にすると学習をスキップして eval のみ実行 |
| `EVAL_SAMPLES` | `50` | 評価サンプル数（`0` = 評価スキップ） |
| `EVAL_SPLIT_RATIO` | `0.1` | dataset に eval split がない場合の train からの切り出し比率 |
| `PUSH_TO_HUB` | — | マージ済みチェックポイントの push 先 HF repo id |
| `OUTPUT_DIR` | `/tmp/lfm2-fox-co` | チェックポイント保存先 |
| `WANDB_PROJECT` | `hack-the-liquid-way` | W&B プロジェクト名 |
| `WANDB_RUN_NAME` | — | W&B run 名 |

### Eval split の解決順序

1. HF dataset に `validation` split がある → そちらを使用
2. HF dataset に `test` split がある → そちらを使用
3. どちらもない → `EVAL_SPLIT_RATIO` で `train` から切り出し

---

## HuggingFace Jobs での実行

> **制約**: HF Jobs は単一スクリプトファイルのみアップロードします。
> `eval.py` と `chat_template.jinja` はコンテナ内で参照できないため、
> **HF Jobs 実行時は `EVAL_SAMPLES` が自動で `0` に強制**されます。
> chat template は下記の手順でインライン化が必要です。

### 事前準備: chat template のインライン化

`train.py` の `load_base_model()` 内にある

```python
template_path = pathlib.Path(__file__).parent / "chat_template.jinja"
tok.chat_template = template_path.read_text(encoding="utf-8")
```

を以下のように書き換えてから submit してください:

```python
tok.chat_template = r"""{{- bos_token -}}
... (chat_template.jinja の内容をそのまま貼り付け) ...
"""
```

### 起動スクリプト（推奨）

`.env` に `HF_TOKEN` / `WANDB_API_KEY` / `WANDB_ENTITY` / `WANDB_PROJECT` を設定した上で:

```bash
# 最小構成
DATASET=your-org/japanese-pii-sft \
  ./experiments/02_training_and_eval/launch_hf_job.sh

# ハイパーパラメータ指定 + モデル push
DATASET=your-org/japanese-pii-sft \
MAX_STEPS=500 \
PUSH_TO_HUB=your-org/fox-co-finetuned \
WANDB_RUN_NAME=fox-co-run-01 \
  ./experiments/02_training_and_eval/launch_hf_job.sh

# dry run（submit せずにゲートと引数を確認）
DRY_RUN=1 \
DATASET=your-org/japanese-pii-sft \
  ./experiments/02_training_and_eval/launch_hf_job.sh
```


スクリプトが自動で行うこと:

- `.env` の読み込みと必須変数のチェック
- `PUSH_TO_HUB` が設定されている場合の HF Token 書き込み権限チェック
- secrets を一時ファイル経由で渡す（argv / `ps aux` に露出させない）
- `EVAL_SAMPLES=0` を強制（HF Jobs コンテナ内で `eval.py` が参照不可のため）

### 起動スクリプトの主な環境変数

| 変数 | デフォルト | 説明 |
|---|---|---|
| `DATASET` | **必須** | HF dataset id |
| `HF_FLAVOR` | `a100-large` | GPU 種別（[料金表](https://huggingface.co/docs/hub/jobs-pricing)） |
| `HF_TIMEOUT` | `2h` | タイムアウト（この時点で課金停止） |
| `HF_NAMESPACE` | — | 組織名前空間（省略時は個人アカウント） |
| `DRY_RUN` | — | `1` にすると submit をスキップしてコマンドを表示 |
| `PUSH_TO_HUB` | — | マージ済みチェックポイントの push 先 HF repo id |
| `WANDB_RUN_NAME` | — | W&B run 名 |
| `MAX_STEPS` / `BATCH_SIZE` / `LR` | train.py デフォルト | ハイパーパラメータ |

### 手動での起動（参考）

```bash
SECRETS=$(mktemp)
cat > "$SECRETS" <<EOF
HF_TOKEN=$HF_TOKEN
WANDB_API_KEY=$WANDB_API_KEY
WANDB_ENTITY=$WANDB_ENTITY
WANDB_PROJECT=$WANDB_PROJECT
EOF

uv run --no-sync hf jobs uv run \
  --flavor a100-large \
  --timeout 2h \
  --secrets-file "$SECRETS" \
  --env DATASET=your-org/japanese-pii-sft \
  --env EVAL_SAMPLES=0 \
  --env MAX_STEPS=500 \
  --env PUSH_TO_HUB=your-org/fox-co-finetuned \
  --env WANDB_RUN_NAME=fox-co-run-01 \
  --detach \
  experiments/02_training_and_eval/train.py

rm -f "$SECRETS"
```

### HF Jobs 後の評価（ローカルで実行）

学習完了後、`PUSH_TO_HUB` で push したモデルをローカルで評価:

```bash
MODEL_ID=your-org/fox-co-finetuned \
DATASET=your-org/japanese-pii-sft \
SKIP_TRAINING=1 \
EVAL_SAMPLES=100 \
  uv run experiments/02_training_and_eval/train.py
```

---

## eval.py — 評価モジュール（import して使う）

`eval.py` は副作用なしのピュア関数を提供します。ノートブックやスクリプトから直接 import できます。

```python
import sys
sys.path.insert(0, "experiments/02_training_and_eval")
from eval import run_eval, log_to_wandb, print_report

# 推論 + メトリクス計算（W&B / print 依存なし）
result = run_eval(model, tok, eval_ds, n_samples=100)

# stdout に F1昇順（弱い順）の表を表示
print_report(result)

# W&B にスカラー + カテゴリ別 Table + bar chart を記録
log_to_wandb(result, wandb_run)
```

### EvalResult の中身

```python
result.n_samples          # 評価サンプル数
result.json_parse_rate    # JSON として parse できた割合
result.micro_f1           # 全カテゴリ合計のマイクロ平均 F1
result.micro_precision    # 同 Precision
result.micro_recall       # 同 Recall
result.per_category       # dict[str, CategoryResult]

result.per_category["financial_info"].f1        # カテゴリ別 F1
result.per_category["financial_info"].tp        # TP 数
```

### W&B に記録されるメトリクス

| キー | 内容 |
|---|---|
| `eval/json_parse_rate` | JSON parse 成功率 |
| `eval/micro_f1` | 全カテゴリ マイクロ平均 F1 |
| `eval/{category}/f1` | カテゴリ別 F1（11カテゴリ） |
| `eval/per_category` | カテゴリ別 Table（F1昇順） |
| `eval/per_category_f1_chart` | bar chart（弱いカテゴリが一目でわかる） |

---

## chat_template.jinja — Chat Template

LFM2 モデルに fox-co PII 抽出用のシステムプロンプトを注入する Jinja2 テンプレートです。

- `train.py` の `load_base_model()` がトークナイザーに自動で注入します
- チェックポイント保存時に `tokenizer_config.json` に埋め込まれます
- system ロールのメッセージを含む場合はデフォルトプロンプトを上書きできます

> **HF Jobs に提出する場合**: HF Jobs は単一ファイルをアップロードするため、
> テンプレート本文を `train.py` 内に文字列定数としてインライン化してください。


