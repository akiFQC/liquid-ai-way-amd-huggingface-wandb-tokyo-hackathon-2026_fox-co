# 機密情報抽出モデル (fox-co PII Extractor)

日本語テキストから社外秘の固有表現を11カテゴリで抽出する LFM2 ベースのモデルです。  
組織内の文書・ログ・メールなどに含まれる機密情報を構造化 JSON として出力します。

## 公開モデル

| モデル | 形式 | 説明 |
|---|---|---|
| [`akiFQC/LFM2.5-1.2B-JP-202606-Conf-Extract`](https://huggingface.co/akiFQC/LFM2.5-1.2B-JP-202606-Conf-Extract) | HF SafeTensors | 学習済みマージチェックポイント |
| [`akiFQC/LFM2.5-1.2B-JP-202606-Conf-Extract-GGUF`](https://huggingface.co/akiFQC/LFM2.5-1.2B-JP-202606-Conf-Extract-GGUF) | GGUF | llama.cpp / on-device 推論向け（Q4_K_M, Q8_0, BF16） |

---

## 出力フォーマット

入力テキストに対して、以下の JSON オブジェクトを1行で出力します。**全 11 キーを必ず含み**、該当なしは空リスト `[]` とします。

### JSON スキーマ

```json
{
  "address":            ["住所・所在地"],
  "company_name":       ["企業・研究機関・組織名"],
  "email_address":      ["メールアドレス"],
  "human_name":         ["人名"],
  "phone_number":       ["電話番号"],
  "account_identifier": ["ユーザーID・アカウント名・従業員番号・マイナンバーなど"],
  "network_identifier": ["IPアドレス・MACアドレス・内部ドメイン・ホスト名"],
  "system_config":      ["ファイルパス・ディレクトリ構造・DBテーブル/カラム名"],
  "project_info":       ["プロジェクト名・開発コードネーム・未発表製品/機能名"],
  "financial_info":     ["売上・原価・利益率・契約金額・個人の給与/報酬額"],
  "transaction_id":     ["契約書番号・請求書番号・見積書番号・顧客管理ID"]
}
```

### カテゴリ定義

| カテゴリ | 定義 | 例 |
|---|---|---|
| `address` | 住所・所在地 | `東京都港区〇〇1-2-3` |
| `company_name` | 企業・研究機関・組織名 | `株式会社サンプル`, `Liquid AI` |
| `email_address` | メールアドレス | `celegans@liquid.ai` |
| `human_name` | 人名（姓名、ニックネームを含む） | `山田 太郎`, `ラミン` |
| `phone_number` | 電話番号（国際番号を含む） | `03-1234-5678`, `+1-800-000-0000` |
| `account_identifier` | アカウント識別子 | `user_89012`, `EMP-9934` |
| `network_identifier` | ネットワーク識別情報 | `192.168.1.15`, `internal-db.local` |
| `system_config` | システム構成情報 | `/var/log/syslog`, `users_table` |
| `project_info` | プロジェクト関連情報 | `Project Apollo`, `次期決済システム開発` |
| `financial_info` | 金額・財務情報 | `1,500,000円`, `$50,000` |
| `transaction_id` | 取引管理番号 | `INV-20260606-001`, `PO-98765` |

### 出力例

入力:

```
本件は山田 太郎（user_89012）から celegans@liquid.ai へ送信されたメールです。
サーバー 192.168.1.15 (/var/log/syslog) の障害について、
Project Apollo の予算 1,500,000円 に関わる契約書番号 INV-20260606-001 を確認してください。
担当: 株式会社サンプル、東京都港区〇〇1-2-3、03-1234-5678
```

出力（1行 JSON）:

```json
{"address": ["東京都港区〇〇1-2-3"], "company_name": ["株式会社サンプル"], "email_address": ["celegans@liquid.ai"], "human_name": ["山田 太郎"], "phone_number": ["03-1234-5678"], "account_identifier": ["user_89012"], "network_identifier": ["192.168.1.15"], "system_config": ["/var/log/syslog"], "project_info": ["Project Apollo"], "financial_info": ["1,500,000円"], "transaction_id": ["INV-20260606-001"]}
```

---

## 使い方

### Python (transformers)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "akiFQC/LFM2.5-1.2B-JP-202606-Conf-Extract"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
model.eval()  # 重要: LoRA 学習後の dropout を無効化

text = "本件は山田 太郎（user_89012）から celegans@liquid.ai へ..."

messages = [{"role": "user", "content": text}]
inputs = tok.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,   # transformers 5.x は必須
).to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.3,
        min_p=0.15,
        repetition_penalty=1.05,
    )

generated = tok.decode(
    output_ids[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)
print(generated)
# → {"address": [...], "company_name": [...], ...}
```

> **Note:** `model.eval()` を必ず呼んでください。`trainer.train()` はドロップアウトを有効にしたまま終了するため、評価時に呼ばないとモデル出力が崩壊します（doom-loop 症状）。

### JSON パース

```python
import json

try:
    entities = json.loads(generated)
    print("人名:", entities["human_name"])
    print("メール:", entities["email_address"])
except json.JSONDecodeError:
    print("JSON parse 失敗（max_new_tokens を増やすか、生成設定を確認）")
```

### llama.cpp (GGUF / on-device)

```bash
# Q4_K_M (Int4) — on-device 推論向け
./llama-cli \
  --model ./LFM2.5-1.2B-JP-202606-Conf-Extract-Q4_K_M.gguf \
  --n-gpu-layers 99 \
  --temp 0.3 --min-p 0.15 --repeat-penalty 1.05 \
  --chat-template lfm2 \
  --interactive --color

# Q8_0 (Int8) — バランス型
./llama-cli \
  --model ./LFM2.5-1.2B-JP-202606-Conf-Extract-Q8_0.gguf \
  --n-gpu-layers 99 \
  --temp 0.3 --min-p 0.15 --repeat-penalty 1.05 \
  --chat-template lfm2 \
  --interactive --color
```

---

## モデル詳細

| 項目 | 値 |
|---|---|
| ベースモデル | `LiquidAI/LFM2.5-1.2B` |
| 学習手法 | LoRA (PEFT) + SFTTrainer (TRL) |
| LoRA rank / alpha | 16 / 32 |
| LoRA ターゲット | `q/k/v/o_proj`, `gate/up/down_proj` |
| 学習率 | 2e-4 |
| Batch size | 4 (per device) × gradient_accumulation 16 = 実効 64 |
| 精度 | bf16 |
| 最大コンテキスト長 | 2048 tokens |
| Optimizer | AdamW (weight_decay=0.01) |
| LR スケジューラ | Linear warmup |
| チャットテンプレート | LFM2 カスタム（fox-co PII 抽出用システムプロンプト注入済み） |

---

## 評価指標

評価は11カテゴリの **文字列完全一致** に基づく Micro F1 で行います。

| 指標 | 内容 |
|---|---|
| `json_parse_rate` | 出力が valid JSON として parse できた割合 |
| `micro_f1` | 全カテゴリ合計のマイクロ平均 F1 |
| `eval/{category}/f1` | カテゴリ別 F1（W&B に記録） |

評価の詳細は [`experiments/02_training_and_eval/eval.py`](02_training_and_eval/eval.py) を参照してください。

---

## 実験ステップ

### Step1: データの準備

[`experiments/01_data_processing/`](01_data_processing/) — PII データセットの収集・前処理・合成データ生成を行います。

### Step2: 学習と評価

[`experiments/02_training_and_eval/`](02_training_and_eval/) — LFM2 LoRA fine-tune + カテゴリ別 F1 評価を行います。

```bash
# HF Jobs で学習実行
DATASET=your-org/japanese-pii-sft \
PUSH_TO_HUB=akiFQC/LFM2.5-1.2B-JP-202606-Conf-Extract \
  ./experiments/02_training_and_eval/launch_hf_job.sh
```

### Step3: GGUF 変換

[`experiments/03_model_format_converting/`](03_model_format_converting/) — 学習済みモデルを GGUF 形式（Q4_K_M / Q8_0 / BF16）に変換して HF Hub へアップロードします。

```bash
# HF Jobs で GGUF 変換実行
./experiments/03_model_format_converting/launch_hf_job.sh

# 変換後: https://huggingface.co/akiFQC/LFM2.5-1.2B-JP-202606-Conf-Extract-GGUF
```

---

## 注意事項・制限

- **出力はテキストのみ**。抽出精度はデータ品質とコンテキスト長に依存します。
- **誤抽出・見逃しが発生します**。高精度が必要なユースケースでは後段の検証ステップを設けてください。
- モデルはルールベースのフィルタリングを置き換えるものではなく、補助ツールとして使用することを想定しています。
- 本モデルは `LiquidAI/LFM2.5-1.2B` をベースとしており、そのライセンスに従います。
