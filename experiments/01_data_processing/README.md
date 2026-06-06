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