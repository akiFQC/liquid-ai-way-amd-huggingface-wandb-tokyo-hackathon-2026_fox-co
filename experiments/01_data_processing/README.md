# データの準備

### 出力データの形式

次のJSON形式の文字列を出力する
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Sensitive Information Entities",
  "description": "社外に秘匿すべき固有表現の抽出結果フォーマット",
  "type": "object",
  "properties": {
    "address": {
      "type": "array",
      "description": "住所／所在地",
      "items": { "type": "string" }
    },
    "company_name": {
      "type": "array",
      "description": "企業／研究機関／組織名",
      "items": { "type": "string" }
    },
    "email_address": {
      "type": "array",
      "description": "メールアドレス",
      "items": { "type": "string" }
    },
    "human_name": {
      "type": "array",
      "description": "人名",
      "items": { "type": "string" }
    },
    "phone_number": {
      "type": "array",
      "description": "電話番号",
      "items": { "type": "string" }
    },
    "account_identifier": {
      "type": "array",
      "description": "アカウント識別子: ユーザーID、アカウント名、従業員番号",
      "items": { "type": "string" }
    },
    "network_identifier": {
      "type": "array",
      "description": "ネットワーク識別情報: IPアドレス、MACアドレス、内部ドメイン・ホスト名",
      "items": { "type": "string" }
    },
    "system_config": {
      "type": "array",
      "description": "システム構成情報: ファイルパス、ディレクトリ構造、DBテーブル・カラム名",
      "items": { "type": "string" }
    },
    "project_info": {
      "type": "array",
      "description": "プロジェクト関連情報: プロジェクト名、開発コードネーム、未発表の製品・機能名",
      "items": { "type": "string" }
    },
    "financial_info": {
      "type": "array",
      "description": "金額・財務情報: 売上、原価、利益率、契約金額、個人の給与・報酬額など",
      "items": { "type": "string" }
    },
    "transaction_id": {
      "type": "array",
      "description": "取引管理番号: 契約書番号、請求書番号、見積書番号、顧客管理ID",
      "items": { "type": "string" }
    }
  },
  "required": [
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
    "transaction_id"
  ],
  "additionalProperties": false
}
```

出力文字列:

```json
{ "address": ["東京都港区〇〇1-2-3"], "company_name": ["株式会社サンプル", "Liquid AI"], "email_address": ["celegans@liquid.ai"], "human_name": ["ラミン", "山田 太郎"], "phone_number": ["010-000-0000", "03-1234-5678"], "account_identifier": ["user_89012", "EMP-9934"], "network_identifier": ["192.168.1.15", "00:1B:44:11:3A:B7", "internal-db.local"], "system_config": ["/var/log/syslog", "users_table", "password_hash"], "project_info": ["Project Apollo", "次期決済システム開発"], "financial_info": ["1,500,000円", "$50,000"], "transaction_id": ["INV-20260606-001", "PO-98765"] }
```


### FOX_COカテゴリ

カテゴリの名前を"FOX_CO"カテゴリと呼称します。

### LFM2-350M-PII-Extract-JPの既存カテゴリ

- 住所／所在地（JSON key: address）
- 企業／研究機関／組織名（JSON key: company_name）
- メールアドレス（JSON key: email_address）
- 人名（JSON key: human_name）
- 電話番号（JSON key: phone_number）

### 新規カテゴリ

- アカウント識別子: ユーザーID、アカウント名、従業員番号、社会保障番号、マイナンバー等公的識別子（JSON key: account_identifier） 
- ネットワーク識別情報: IPアドレス（特にローカルIP）MACアドレス、内部ドメイン・ホスト名（JSON key: network_identifier） 
- システム構成情報: ファイルパス、ディレクトリ構造、データベースのテーブル・カラム名、エラーログの一部（JSON key: system_config） 
- プロジェクト関連情報: プロジェクト名、開発コードネーム、未発表の製品・機能名（JSON key: project_info）
- 金額・財務情報: 売上、原価、利益率、契約金額、個人の給与・報酬額など（JSON key: financial_info） 
- 取引管理番号: 契約書番号、請求書番号、見積書番号、顧客管理ID（JSON key: transaction_id）

### カテゴリ対応表


| FOX_COカテゴリ (JSON key) | [OpenPII 1.5M](https://www.google.com/search?q=ai4privacy/pii-masking-openpii-1.5m) の該当ラベル | [ner-wikipedia-dataset](https://github.com/stockmark/ner-wikipedia-dataset) の該当ラベル |
| --- | --- | --- |
| 住所／所在地 `address` | `CITY` `STREETBUILDINGNUM` `ZIPCODE` | `地名` `施設名` |
| 企業／研究機関／組織名 `company_name` | `TITLE`（一部関連） | `法人名` `政治的組織名` `その他の組織名` |
| メールアドレス `email_address` | `EMAIL` | （該当なし） |
| 人名 `human_name` | `GIVENNAME` `SURNAME` | `人名` |
| 電話番号 `phone_number` | `TELEPHONENUM` | （該当なし） |
| アカウント識別子・公的識別子 `account_identifier` | `USERNAME` `IDCARDNUM` `DRIVERLICENSENUM` `SOCIALNUM` | （該当なし） |
| ネットワーク識別情報 `network_identifier` | （該当なし） | （該当なし） |
| システム構成情報 `system_config` | （該当なし） | （該当なし） |
| プロジェクト関連情報 `project_info` | （該当なし） | `製品名` |
| 金額・財務情報 `financial_info` | `CREDITCARDNUMBER` `TAXNUM` | （該当なし） |
| 取引管理番号 `transaction_id` | （該当なし） | （該当なし） |

## 処理ステップ

### 1. 上記で指定したJSON keyのカテゴリに合わせて、OpenPII 1.5Mとer-wikipedia-datasetそれぞれを整形して、上記のJSON schemaの形式に変換する

- データ形式としては、`input_text`, `annotation_json` のカラムを持つこと
- huggingface datasetsの形式にしておく事。
- 変換後は、適当な名前で `experiments/data` に一時的に保存しておくこと。

#### 保存形式

HuggingFace `datasets` の Arrow 形式（`Dataset.save_to_disk()`）で保存する。各行は以下の2カラムを持つ。

| カラム名 | 型 | 内容 |
| --- | --- | --- |
| `input_text` | `string` | PIIを含む元テキスト |
| `annotation_json` | `string` | FOX_COカテゴリ11キーをすべて含むJSONオブジェクトの文字列。対応エンティティが存在しないキーは空リスト `[]` |

行の例:

```json
{
  "input_text": "田中一郎さんへ、請求書をtanaka@example.co.jpまでお送りください。",
  "annotation_json": "{\"address\": [], \"company_name\": [], \"email_address\": [\"tanaka@example.co.jp\"], \"human_name\": [\"田中一郎\"], \"phone_number\": [], \"account_identifier\": [], \"network_identifier\": [], \"system_config\": [], \"project_info\": [], \"financial_info\": [], \"transaction_id\": []}"
}
```

#### 使用方法

##### スクリプトの実行

```bash
# リポジトリルートから実行
uv run experiments/01_data_processing/prepare_datasets.py
```

HuggingFace Hub からデータセットをダウンロードし、変換後に以下へ保存する:

| 保存先 | 元データセット |
| --- | --- |
| `experiments/data/openpii_processed/` | `ai4privacy/pii-masking-openpii-1.5m` |
| `experiments/data/ner_wikipedia_processed/` | `stockmark/ner-wikipedia-dataset` |

##### 出力データの確認

```python
from datasets import load_from_disk
import json

ds = load_from_disk("experiments/data/openpii_processed")
row = ds[0]
print(row["input_text"])
print(json.loads(row["annotation_json"]))
# → {"address": [...], "company_name": [...], ..., "transaction_id": []}
```

##### 注意事項

- `network_identifier` / `system_config` / `transaction_id` は両データセットに対応ラベルがないため、常に空リスト `[]` となる。
- OpenPII 1.5M は多言語データセット（約150万行）だが、`language == "Japanese"` の行のみ抽出して使用する。ダウンロード自体は全言語分が対象となるため時間がかかる。

##### 結果
```

[OpenPII 1.5M (ja)] 20,754 rows
  カテゴリ                                  件数        行数       行カバー率
  ------------------------------  --------  --------  ----------
  address                           17,082    11,646       56.1%
  company_name                      10,309     9,362       45.1%
  email_address                     11,147    10,416       50.2%
  human_name                        32,248    14,775       71.2%
  phone_number                       8,472     8,054       38.8%
  account_identifier                12,956     8,496       40.9%
  network_identifier                     0         0        0.0%
  system_config                          0         0        0.0%
  project_info                           0         0        0.0%
  financial_info                     9,399     7,396       35.6%
  transaction_id                         0         0        0.0%
  

[ner-wikipedia-dataset] 5,343 rows
  カテゴリ                                  件数        行数       行カバー率
  ------------------------------  --------  --------  ----------
  address                            3,265     1,891       35.4%
  company_name                       4,716     3,015       56.4%
  email_address                          0         0        0.0%
  human_name                         2,980     1,739       32.5%
  phone_number                           0         0        0.0%
  account_identifier                     0         0        0.0%
  network_identifier                     0         0        0.0%
  system_config                          0         0        0.0%
  project_info                       1,215       848       15.9%
  financial_info                         0         0        0.0%
  transaction_id                         0         0        0.0%
```

### 2. カバレッジ不足カテゴリの整理と合成データ準備

ステップ1の変換結果（統計）から、以下のカテゴリが既存データセットではカバーできていないことが判明した。

| FOX_COカテゴリ | OpenPII (ja) | ner-wikipedia | 状態 |
| --- | --- | --- | --- |
| `network_identifier` | 0件 | 0件 | **完全未カバー** → 合成必要 |
| `system_config` | 0件 | 0件 | **完全未カバー** → 合成必要 |
| `financial_info` | 9,399件 ※1 | 0件 | **部分カバー** → 不足分を合成 |
| `transaction_id` | 0件 | 0件 | **完全未カバー** → 合成必要 |

※1 OpenPII の `financial_info` は `CREDITCARDNUMBER`・`TAXNUM` のみ。`売上`・`原価`・`利益率`・`契約金額`・`個人の給与・報酬額` は未カバー。

データ合成の前に、各カテゴリのサブタイプ・発生シチュエーション・例を以下のYAMLに整理する。

```
experiments/01_data_processing/category_breakdown.yaml
```

### 3. OSSなLLMを用いたデータ合成

`experiments/01_data_processing/category_breakdown.yaml`でブレイクダンしたそれぞれのデータに対して次のような流れでデータ合成を行う

疑似コード
```python
positive_data = []
negative_data = []
for category in all_categories:
  for subcategory in category["list_subcategory"]:
    for situatiation in category["list_situatiation"]:
      text_with_confidential_info = LLM("{situatiation}という場面で{subcategory}を含む例文を作成して")
      positive_data.append(text_with_confidential_info)

      text_outwith_confidential_info = LLM("{situatiation}という場面で{subcategory}を含みそうでギリギリ含まないような例文を作成して")
      negative_data.append(text_outwith_confidential_info)

dataset= positive_data + positive_data
```
これはあくまで例です。実際にはfew-shot promptingなどで例を与えたりしています。

データ合成には`src/utils/hf_llm_inference.py`をモジュールとして使います。

#### 実装方針

注釈（`annotation_json`）の正確性を担保するため、LLM に注釈 JSON を直接生成させるのではなく、

> **先に entity 値をプログラムで合成し、LLM には「その値を verbatim で含む日本語例文」だけを書かせる**

二段構えにしている（`synthesize_data.py` の `ENTITY_GENERATORS`）。

- **positive**: サブタイプ形式に沿ったランダムな現実的値（IP/MAC/請求書番号/金額など）を生成 → LLM がその値を一字一句そのまま含む日本語例文を作成 → 生成例文に値が verbatim で含まれることを検証してから、`annotation_json` の **対象カテゴリのみ**に値を格納（指定値以外の機密情報は入れないよう指示）。
- **negative**: 対象カテゴリの話題に触れつつ具体値を含まないハードネガティブ例文を生成 → カテゴリ単位の正規表現で値の漏れを best-effort 検査 → `annotation_json` は **全11キー空配列**。

採用モデルは **`google/gemma-4-26B-A4B-it`**（`deepinfra` プロバイダ経由、コスパ・ライセンス自由度重視）。`MODEL` / `PROVIDER` で上書き可。

#### 使用方法

```bash
# リポジトリルートから実行（.env の HF_TOKEN を自動読込）
uv run experiments/01_data_processing/synthesize_data.py

# 動作確認: 各サブタイプ1シチュエーションだけ・並列度を抑えて少量生成
uv run experiments/01_data_processing/synthesize_data.py --limit-situations 1 --concurrency 8

# モデル / 並列度の上書き
MODEL=... PROVIDER=... uv run experiments/01_data_processing/synthesize_data.py --concurrency 24
```

生成結果は1件完了するごとに（tqdm の進捗バー付きで）逐次保存される。

| 保存先 | 内容 |
| --- | --- |
| `experiments/data/synthetic/synthetic.jsonl` | 逐次追記される生データ（`input_text` / `annotation_json` + provenance 用 `_meta`）。中断後の再実行で `_meta.task_id` を見て resume する |
| `experiments/data/synthetic_processed/` | 完了後に JSONL を変換した HF `datasets` の Arrow 形式（`input_text` / `annotation_json` の2カラムのみ）。`--no-export` で抑止 |

主なオプション: `--reps`（同一組み合わせの生成本数）、`--limit-situations`、`--temperature`、`--seed`（値生成の再現）、`--overwrite`（resume せず作り直す）。


###

### 4. データの混合とデータ拡張

1と3で作成したデータを混合する。

#### 3つのデータの統計上の整理





このとき、
```
```

###　5. データ整形

###　6. 