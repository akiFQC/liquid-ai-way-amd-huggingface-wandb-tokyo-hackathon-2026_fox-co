# 機密情報を検知可能なLFMの学習

この実験では、組織における機密情報を抽出可能なLLMを学習します。

## 出力のラベル設計

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

## Step1: データの準備

`experiments_/01_data_processing` 


## Step2: 学習と評価

`experiments/02_training_and_eval` 


## Step3: GGUFへの変換

`akiFQC/LFM2.5-1.2B-JP-202606-Conf-Extract`をGGUFフォーマットに変換して、HFにuploadしなおしますます。



`experiments/03_model_format_converting` 

アップロード先: `akiFQC/LFM2.5-1.2B-JP-202606-Conf-Extract-GGUF`