"""
OSS LLM によるFOX_COカテゴリの合成データ生成（README 処理ステップ3）。

`category_breakdown.yaml` でブレイクダウンした未カバーカテゴリ
(network_identifier / system_config / financial_info / transaction_id) に対し、
カテゴリ × サブタイプ × シチュエーションの組み合わせごとに

  - positive : その秘匿情報を「含む」日本語例文
  - negative : 含みそうでギリギリ「含まない」日本語例文（ハードネガティブ）

を生成する。

注釈（annotation）の正確性を担保するため、LLM に annotation JSON を直接吐かせるのではなく、
**先に entity 値をプログラムで合成し、LLM には「その値を verbatim で含む日本語例文」だけを書かせる**
二段構えにする。これにより ground truth が完全に正確になる。

生成結果は、

  {"input_text": "...", "annotation_json": "{...11キー...}"}

の形式（README 準拠）で JSONL に逐次（1件完了するごとに）追記保存する。tqdm で進捗表示。
途中再開（resume）にも対応する。

モデルは **コストパフォーマンスとライセンス自由度** を両立する
`google/gemma-4-26B-A4B-it` を既定とする（provider は `auto` で利用可能なプロバイダを自動選択）。

推論インフラは `src/utils/hf_llm_inference.py` をモジュールとして利用する。

使い方:
    # リポジトリルートから（.env の HF_TOKEN を自動読込）
    uv run experiments/01_data_processing/synthesize_data.py

    # 動作確認（各サブタイプ1シチュエーションだけ・少量）
    uv run experiments/01_data_processing/synthesize_data.py --limit-situations 1

    # モデル・並列度の上書き
    MODEL=... PROVIDER=... uv run experiments/01_data_processing/synthesize_data.py --concurrency 24

完了後、JSONL を HF datasets の Arrow 形式へ変換して
`experiments/data/synthetic_processed/` に保存する（--no-export で抑止）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Callable

import yaml
from tqdm import tqdm

# src/ を import パスに通して推論インフラを読み込む
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.hf_llm_inference import (  # noqa: E402
    HFInferenceClient,
    InferenceConfig,
    Message,
)

# ---------------------------------------------------------------------------
# スキーマ定義（README 出力データの形式と一致させる）
# ---------------------------------------------------------------------------

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


def _empty_annotation() -> dict[str, list[str]]:
    return {k: [] for k in ALL_KEYS}


# 合成対象（既存データセットで未カバー）の4カテゴリ
TARGET_CATEGORIES = [
    "network_identifier",
    "system_config",
    "financial_info",
    "transaction_id",
]


# ---------------------------------------------------------------------------
# entity 値ジェネレータ（stdlib random のみ）
#
# category_breakdown.yaml の examples を「形式テンプレート」として踏襲し、
# ランダムな現実的値を生成する。値は例文へ verbatim 埋め込み・一致判定が容易な
# クリーンな部分文字列になるようにする（フレーズ型は短いトークンに正規化）。
# ---------------------------------------------------------------------------

_rint = random.randint
_choice = random.choice


def _ymd() -> str:
    return f"2026{_rint(1, 12):02d}{_rint(1, 28):02d}"


def _ym() -> str:
    return f"2026-{_rint(1, 12):02d}"


# --- network_identifier ---------------------------------------------------

def _gen_local_ip_v4() -> str:
    block = _choice(["192.168", "10", "172"])
    if block == "192.168":
        return f"192.168.{_rint(0, 255)}.{_rint(1, 254)}"
    if block == "10":
        return f"10.{_rint(0, 255)}.{_rint(0, 255)}.{_rint(1, 254)}"
    return f"172.{_rint(16, 31)}.{_rint(0, 255)}.{_rint(1, 254)}"


def _gen_local_ip_v6() -> str:
    seg = lambda: f"{_rint(0, 0xFFFF):x}"  # noqa: E731
    prefix = _choice(["fe80", "fd00", f"fd{_rint(0, 0xFF):02x}"])
    return f"{prefix}::{seg()}:{seg()}"


def _gen_subnet_cidr() -> str:
    return _choice([
        f"192.168.{_rint(0, 255)}.0/24",
        "10.0.0.0/8",
        f"172.16.{_rint(0, 255)}.0/22",
        f"10.{_rint(0, 255)}.0.0/16",
    ])


def _gen_mac_address() -> str:
    octets = [f"{_rint(0, 255):02X}" for _ in range(6)]
    sep = _choice([":", "-"])
    mac = sep.join(octets)
    return mac.lower() if random.random() < 0.3 else mac


def _gen_internal_hostname() -> str:
    role = _choice([
        "dev-server", "fileserver", "db-primary", "ws", "app-node",
        "web", "cache", "build", "ci", "gitlab-runner",
    ])
    suffix = _choice([f"-{_rint(1, 99):02d}", f"{_rint(1, 99):02d}", "-prod", "-stg", "-dev"])
    return f"{role}{suffix}"


def _gen_internal_domain() -> str:
    host = _choice([
        "internal-db", "prod-api", "jenkins", "gitlab",
        "confluence", "grafana", "vault", "registry",
    ])
    tld = _choice([".local", ".intranet", ".internal", ".corp.example.com", ".intranet.example.co.jp"])
    return f"{host}{tld}"


def _gen_internal_url() -> str:
    scheme = _choice(["http", "https"])
    host = _choice([_gen_internal_domain(), _gen_local_ip_v4()])
    port = _choice(["", f":{_choice([8080, 8443, 3000, 9000])}"])
    path = _choice(["/admin", "/hr/portal", "/display/DEV", "/d/api-latency", "/dashboard", "/api/internal/health"])
    return f"{scheme}://{host}{port}{path}"


def _gen_network_share_path() -> str:
    server = _choice(["fileserver01", "nas.local", "fileserver02", _gen_local_ip_v4()])
    share = _choice(["共有フォルダ", "backup", "documents", "project", "共有"])
    if random.random() < 0.5:
        return f"\\\\{server}\\{share}"
    return f"smb://{server}/{share}"


def _gen_port_service() -> str:
    port = _choice([5432, 6379, 8443, 3306, 27017, 9092, 19876, 50051])
    return _choice([f"ポート {port}", f":{port}", f"{port}番ポート", f"TCP/{port}"])


def _gen_vpn_wifi_config() -> str:
    kind = _choice(["vpn_host", "ssid", "endpoint", "gw"])
    if kind == "vpn_host":
        return _choice(["vpn.corp.example.co.jp", "vpn.example-vpn.local", "vpn.internal"])
    if kind == "ssid":
        return _choice(["EXAMPLE-CORP-5G", "CORP-WIFI", "OFFICE-SECURE-5G"])
    if kind == "endpoint":
        return f"{_gen_local_ip_v4()}:{_choice([51820, 1194, 500])}"
    return _choice(["gw.example-vpn.local", "gw.corp.internal"])


# --- system_config --------------------------------------------------------

def _gen_file_path_unix() -> str:
    d = _choice([
        "/var/log/app", "/home/deploy/releases", "/etc/nginx/sites-enabled",
        "/opt/app/src/services", "/srv/app/config", "/opt/airflow/dags",
    ])
    f = _choice([
        "error.log", "database.yml", "internal-portal.conf", "PaymentService.java",
        "secrets.yml", "etl_pipeline.py", "app.log", "settings.py",
    ])
    return f"{d}/{f}"


def _gen_file_path_windows() -> str:
    return _choice([
        r"C:\inetpub\wwwroot\api\Web.config",
        r"C:\Users\Administrator\AppData\Local\Temp\crash.dmp",
        r"D:\deploy\app\bin\Release\net8.0\appsettings.json",
        r"HKLM\SOFTWARE\ExampleCorp\LicenseKey",
        rf"C:\Program Files\App\config{_rint(1, 9)}.ini",
    ])


def _gen_db_schema_table() -> str:
    schema = _choice(["schema_production", "hr_db", "analytics", "dw_staging", "sales_db", "core"])
    table = _choice(["financial_records", "employees", "raw_events", "fact_sales_daily", "orders", "payments", "users"])
    return f"{schema}.{table}"


def _gen_db_column() -> str:
    return _choice([
        "password_hash", "auth_token", "salary_amount",
        "users.password_hash", "orders.customer_id",
        "hr_db.employees.salary_amount", "sessions.auth_token",
    ])


def _gen_error_log_stacktrace() -> str:
    return _choice([
        f"NullPointerException at com.example.internal.UserRepository.findById(UserRepository.java:{_rint(10, 300)})",
        f"ERROR in /opt/app/src/services/PaymentService.java:{_rint(10, 300)}",
        f'File "/opt/airflow/dags/etl_pipeline.py", line {_rint(10, 200)}, in run',
        "[FATAL] /srv/app/config/secrets.yml: permission denied",
    ])


def _gen_env_variable() -> str:
    return _choice([
        "DATABASE_URL", "SECRET_KEY", "AWS_ACCESS_KEY_ID",
        "INTERNAL_API_BASE_URL=http://api.internal:8080",
        "REDIS_URL", "JWT_SECRET", "SENTRY_DSN",
    ])


def _gen_config_key_value() -> str:
    return _choice([
        f"max_connections: {_choice([100, 200, 500])}",
        "redis_host: cache.internal, redis_port: 6379",
        "[database] host=db-primary.local user=app_user",
        f"worker_processes: {_rint(2, 8)}, timeout: {_choice([30, 60, 120])}",
    ])


def _gen_container_image() -> str:
    reg = _choice(["registry.internal", "docker.corp.example.com", "ghcr.io/example-corp"])
    name = _choice(["backend/api-server", "ml-pipeline", "internal-tool", "frontend/web"])
    tag = _choice([
        f"v{_rint(1, 5)}.{_rint(0, 9)}.{_rint(0, 9)}",
        f"{_ymd()}-abc{_rint(100, 999)}",
        "latest",
    ])
    return f"{reg}/{name}:{tag}"


def _gen_internal_api_endpoint() -> str:
    return _choice([
        f"http://payment-service.internal:{_choice([8080, 8443])}/v1/charge",
        "grpc://user-service.default.svc.cluster.local:50051",
        f"http://{_gen_local_ip_v4()}:3000/api/internal/health",
    ])


def _gen_cron_batch_schedule() -> str:
    return _choice([
        "0 2 * * * /opt/scripts/db_backup.sh",
        "*/5 * * * * /usr/local/bin/health_check.sh",
        f"{_rint(0, 59)} {_rint(0, 23)} * * * /home/batch/run_etl.py",
    ])


# --- financial_info（金額・率は「裸のトークン」で返す） -------------------

def _man_yen() -> str:
    return f"{_rint(100, 9999):,}万円"


def _oku_man() -> str:
    oku = _rint(1, 9)
    man = _rint(0, 9999)
    return f"{oku}億円" if man == 0 else f"{oku}億{man:,}万円"


def _yen() -> str:
    return f"¥{_rint(1, 9) * 10 ** _rint(5, 8):,}"


def _en() -> str:
    return f"{_rint(1, 99) * 100000:,}円"


def _gen_revenue_sales() -> str:
    return _choice([_oku_man(), _yen(), _man_yen()])


def _gen_cost_cogs() -> str:
    return _choice([_man_yen(), _yen(), f"{_rint(40, 90)}%"])


def _gen_profit_margin() -> str:
    return _choice([f"{_rint(5, 40)}.{_rint(0, 9)}%", f"{_rint(5, 40)}%", _man_yen()])


def _gen_budget_forecast() -> str:
    return _choice([_oku_man(), _man_yen(), _yen()])


def _gen_contract_amount() -> str:
    return _choice([_en(), f"${_rint(10, 500) * 1000:,} USD", _yen()])


def _gen_investment_funding() -> str:
    return _choice([f"{_rint(1, 50)}億円", f"${_rint(1, 20)}M USD", _oku_man()])


def _gen_salary_compensation() -> str:
    return _choice([f"{_rint(25, 120)}万円", f"{_rint(20, 80) * 10000:,}円", f"年俸 {_rint(400, 1500)}万円"])


def _gen_freelance_fee() -> str:
    return _choice([f"{_rint(30, 120) * 10000:,}円", f"時間単価 ¥{_rint(8, 30) * 1000:,}", f"{_rint(30, 120)}万円"])


def _gen_penalty_settlement() -> str:
    return _choice([_yen(), _man_yen(), f"日額 {_rint(1, 50) * 1000:,}円"])


def _gen_commission_fee() -> str:
    return _choice([f"{_rint(1, 30)}%", _yen(), f"¥{_rint(1, 99) * 1000:,}"])


# --- transaction_id -------------------------------------------------------

def _gen_invoice_number() -> str:
    return _choice([
        f"INV-{_ymd()}-{_rint(1, 999):03d}",
        f"{_ym()}-{_rint(1, 99):04d}",
        f"JP-2026-{_rint(1, 99999):05d}",
        f"BI-2026-{_rint(1, 9999):04d}",
    ])


def _gen_contract_number() -> str:
    return _choice([
        f"CTR-2026-{_rint(1, 9999):04d}",
        f"BCA-JP-2025-{_rint(1, 99):04d}",
        f"SOW-2026-{_rint(1, 999):03d}",
        f"MK-2026-{_rint(1, 9999):04d}",
    ])


def _gen_purchase_order_number() -> str:
    return _choice([
        f"PO-{_rint(10000, 99999)}",
        f"PR-2026-{_rint(1, 12):02d}-{_rint(1, 9999):04d}",
        f"REQ-2026-{_rint(1, 9999):04d}",
        f"PO-{_ymd()}-JP-{_rint(1, 99):04d}",
    ])


def _gen_quotation_number() -> str:
    return _choice([
        f"QT-2026-{_rint(1, 9999):04d}",
        f"PROP-2026-{_rint(1, 9999):04d}",
        f"EST-{_ymd()}-{_rint(1, 999):03d}",
        f"JP2026-Q-{_rint(1, 9999):04d}",
    ])


def _gen_customer_id() -> str:
    return _choice([
        f"CUST-{_rint(1, 99999):05d}",
        f"TRD-JP-{_rint(1, 9999)}",
        f"C-{_ymd()}-{_rint(1, 9999):04d}",
        f"得意先コード {_rint(10000, 99999)}",
    ])


def _gen_project_case_code() -> str:
    return _choice([
        f"PROJ-2026-JP-{_rint(1, 999):04d}",
        f"W-2026-{_rint(1, 9999):04d}",
        "DX-PHASE2",
        f"ORD-{_ymd()}-{_rint(1, 9999):04d}",
    ])


def _gen_approval_number() -> str:
    return _choice([
        f"RNG-2026-{_rint(1, 9999):04d}",
        f"APR-{_ymd()}-{_rint(1, 9999):04d}",
        f"PA-2026-{_rint(1, 9999):04d}",
        f"EXP-2026-{_rint(1, 9999):04d}",
    ])


def _gen_shipment_tracking() -> str:
    return _choice([
        f"{_rint(1000, 9999)}-{_rint(1000, 9999)}-{_rint(1000, 9999)}",
        f"SHP-{_ymd()}-{_rint(1, 9999):04d}",
        f"JP{_rint(10 ** 9, 10 ** 10 - 1)}JP",
        f"WH-IN-2026-{_rint(1, 9999):04d}",
    ])


def _gen_bank_payment_reference() -> str:
    return _choice([
        f"TRF-{_ymd()}-{_rint(1, 9999):04d}",
        f"PMT-2026-{_rint(1, 9999):04d}",
        f"REF: JP{_ymd()}{_rint(1, 9999):04d}",
        f"AAAABB2L{_ymd()}{_rint(100000, 999999)}",
    ])


def _gen_support_ticket() -> str:
    return _choice([
        f"TKT-2026-{_rint(1, 99999):05d}",
        f"INC-{_ymd()}-{_rint(1, 9999):04d}",
        f"SP-2026-JP-{_rint(1, 9999):04d}",
        f"#{_rint(10000, 99999)}",
    ])


ENTITY_GENERATORS: dict[str, Callable[[], str]] = {
    # network_identifier
    "local_ip_v4": _gen_local_ip_v4,
    "local_ip_v6": _gen_local_ip_v6,
    "subnet_cidr": _gen_subnet_cidr,
    "mac_address": _gen_mac_address,
    "internal_hostname": _gen_internal_hostname,
    "internal_domain": _gen_internal_domain,
    "internal_url": _gen_internal_url,
    "network_share_path": _gen_network_share_path,
    "port_service": _gen_port_service,
    "vpn_wifi_config": _gen_vpn_wifi_config,
    # system_config
    "file_path_unix": _gen_file_path_unix,
    "file_path_windows": _gen_file_path_windows,
    "db_schema_table": _gen_db_schema_table,
    "db_column": _gen_db_column,
    "error_log_stacktrace": _gen_error_log_stacktrace,
    "env_variable": _gen_env_variable,
    "config_key_value": _gen_config_key_value,
    "container_image": _gen_container_image,
    "internal_api_endpoint": _gen_internal_api_endpoint,
    "cron_batch_schedule": _gen_cron_batch_schedule,
    # financial_info
    "revenue_sales": _gen_revenue_sales,
    "cost_cogs": _gen_cost_cogs,
    "profit_margin": _gen_profit_margin,
    "budget_forecast": _gen_budget_forecast,
    "contract_amount": _gen_contract_amount,
    "investment_funding": _gen_investment_funding,
    "salary_compensation": _gen_salary_compensation,
    "freelance_fee": _gen_freelance_fee,
    "penalty_settlement": _gen_penalty_settlement,
    "commission_fee": _gen_commission_fee,
    # transaction_id
    "invoice_number": _gen_invoice_number,
    "contract_number": _gen_contract_number,
    "purchase_order_number": _gen_purchase_order_number,
    "quotation_number": _gen_quotation_number,
    "customer_id": _gen_customer_id,
    "project_case_code": _gen_project_case_code,
    "approval_number": _gen_approval_number,
    "shipment_tracking": _gen_shipment_tracking,
    "bank_payment_reference": _gen_bank_payment_reference,
    "support_ticket": _gen_support_ticket,
}

# ---------------------------------------------------------------------------
# LLM による entity 値プール生成（固定リストでは多様性が低いサブタイプ向け）
#
# 起動時に build_entity_pools() で一括生成し、patch_entity_generators() で
# ENTITY_GENERATORS を上書きすることで gen_entities() から透過的に使われる。
# 失敗したサブタイプは元の固定リストジェネレータにフォールバックする。
# ---------------------------------------------------------------------------

LLM_POOL_SUBTYPES: dict[str, dict] = {
    "internal_api_endpoint": {
        "desc": "マイクロサービス間通信・内部APIのエンドポイントURL（外部公開なし、HTTP/gRPC/REST）",
        "examples": [
            "http://payment-service.internal:8080/v1/charge",
            "grpc://user-service.default.svc.cluster.local:50051",
            "http://10.0.1.25:3000/api/internal/health",
        ],
    },
    "cron_batch_schedule": {
        "desc": "cronジョブ・バッチ処理のスケジュール定義（cron式）とスクリプトパスの組み合わせ",
        "examples": [
            "0 2 * * * /opt/scripts/db_backup.sh",
            "毎日03:00に /home/batch/run_etl.py を実行",
            "*/5 * * * * /usr/local/bin/health_check.sh >> /var/log/cron.log",
        ],
    },
    "error_log_stacktrace": {
        "desc": "スタックトレース・エラーログの断片（内部クラス名・パス・行番号を含む1行のログ行）",
        "examples": [
            "ERROR in /opt/app/src/services/PaymentService.java:142",
            "NullPointerException at com.example.internal.UserRepository.findById(UserRepository.java:89)",
            "[FATAL] /srv/app/config/secrets.yml: permission denied",
            'Traceback: File "/opt/airflow/dags/etl_pipeline.py", line 74, in run',
        ],
    },
    "config_key_value": {
        "desc": "設定ファイル（YAML/TOML/INI等）のキーと値の組み合わせ（1行または短いスニペット）",
        "examples": [
            "max_connections: 200",
            "redis_host: cache.internal, redis_port: 6379",
            "[database] host=db-primary.local user=app_user",
            "worker_processes: 4, timeout: 30",
        ],
    },
    "file_path_windows": {
        "desc": "Windows サーバー・PC上の絶対パス（設定ファイル・ログ・IIS・レジストリ等）",
        "examples": [
            r"C:\inetpub\wwwroot\api\Web.config",
            r"C:\Users\Administrator\AppData\Local\Temp\crash.dmp",
            r"D:\deploy\app\bin\Release\net8.0\appsettings.json",
            r"HKLM\SOFTWARE\ExampleCorp\LicenseKey",
        ],
    },
    "db_column": {
        "desc": "データベースのカラム名（特に機密フィールド名、テーブル名.カラム名形式も可）",
        "examples": [
            "users テーブルの password_hash カラム",
            "orders.customer_id と payments.amount を JOIN",
            "hr_db.employees.salary_amount",
            "sessions.auth_token カラム",
        ],
    },
    "env_variable": {
        "desc": "環境変数名・その値の断片（アプリケーション設定・シークレット参照を含む）",
        "examples": [
            "DATABASE_URL 環境変数が未設定",
            "SECRET_KEY を .env から読み込み",
            "AWS_ACCESS_KEY_ID は SSM Parameter Store に格納",
            "INTERNAL_API_BASE_URL=http://api.internal:8080",
        ],
    },
    "vpn_wifi_config": {
        "desc": "VPN接続先・社内WiFi SSID など、内部アクセスに必要なネットワーク識別情報",
        "examples": [
            "VPNサーバー: vpn.corp.example.co.jp",
            "社内WiFi SSID: EXAMPLE-CORP-5G",
            "WireGuardエンドポイント: 203.0.113.10:51820",
            "IPsec ゲートウェイ: gw.example-vpn.local",
        ],
    },
}

_POOL_N = 30  # 1サブタイプあたりの entity 値プールサイズ

_POOL_SYSTEM_PROMPT = (
    "あなたはITセキュリティ・個人情報保護のテストデータ専門家です。"
    "指定された種類の値を、多様かつ現実的に生成します。"
    "1行に1値を出力し、値のみを書いてください（番号・記号・引用符・説明は不要）。"
    "日本の企業・ITシステムで実際に使われそうな値にしてください。"
)


def _build_pool_prompt(meta: dict, n: int) -> str:
    shots = "\n".join(f"- {e}" for e in meta["examples"])
    return (
        f"次の種類の値を {n} 個、なるべく多様に生成してください。\n\n"
        f"種類: {meta['desc']}\n\n"
        f"参考例（これらとは異なる新しい値を生成してください）:\n{shots}\n\n"
        f"1行に1値、値のみを出力してください。"
    )


async def build_entity_pools(
    client: HFInferenceClient,
    cfg: InferenceConfig,
    *,
    n: int = _POOL_N,
    concurrency: int = 8,
) -> dict[str, list[str]]:
    """
    LLM_POOL_SUBTYPES に登録されたサブタイプごとに entity 値プールを LLM で生成する。

    並列実行し、各応答を行単位で分割・クリーニングしてプールを構築する。
    生成失敗・プールが空の場合は空リストを返し、元の固定リストジェネレータにフォールバックする。
    """
    pool_cfg = InferenceConfig(
        model=cfg.model,
        provider=cfg.provider,
        max_tokens=n * 60,   # 1値あたり平均 ~60 token を余裕を持って確保
        temperature=0.9,      # 多様性重視
        top_p=0.95,
        max_retries=cfg.max_retries,
        retry_delay=cfg.retry_delay,
        timeout=cfg.timeout,
    )
    semaphore = asyncio.Semaphore(concurrency)
    subtypes = list(LLM_POOL_SUBTYPES.items())

    print(f"\nLLM entity プールを生成中 ({len(subtypes)} サブタイプ) …")
    pbar = tqdm(total=len(subtypes), desc="entity pools", dynamic_ncols=True)

    async def _fetch(subtype_name: str, meta: dict) -> tuple[str, list[str]]:
        msgs: list[Message] = [
            {"role": "system", "content": _POOL_SYSTEM_PROMPT},
            {"role": "user", "content": _build_pool_prompt(meta, n)},
        ]
        try:
            result = await client.async_chat_complete(msgs, config_override=pool_cfg, semaphore=semaphore)
            lines = []
            for raw in result.content.splitlines():
                # 先頭の番号・箇条書き記号を除去
                ln = re.sub(r'^[\s\-•・*\d\.）)）、]+', '', raw).strip()
                if ln:
                    lines.append(ln)
            pool = list(dict.fromkeys(lines))  # 順序保持で重複排除
            pbar.update(1)
            return subtype_name, pool
        except Exception as exc:  # noqa: BLE001
            tqdm.write(f"[pool error] {subtype_name}: {exc}")
            pbar.update(1)
            return subtype_name, []

    pairs = await asyncio.gather(*(_fetch(name, meta) for name, meta in subtypes))
    pbar.close()

    pools: dict[str, list[str]] = {}
    for name, pool in pairs:
        if pool:
            pools[name] = pool
            print(f"  {name}: {len(pool)} 件生成")
        else:
            print(f"  {name}: 生成失敗 → 固定リストにフォールバック")
    return pools


def patch_entity_generators(pools: dict[str, list[str]]) -> None:
    """プール生成結果で ENTITY_GENERATORS を in-place 上書きする。"""
    for name, pool in pools.items():
        if pool and name in ENTITY_GENERATORS:
            ENTITY_GENERATORS[name] = lambda p=pool: random.choice(p)


def gen_entities(subtype_name: str) -> list[str]:
    """サブタイプに対応する entity 値を 1〜2 個生成する（重複排除）。"""
    gen = ENTITY_GENERATORS[subtype_name]
    n = 1 if random.random() < 0.7 else 2
    vals: list[str] = []
    for _ in range(n * 2):  # 重複で取りこぼさないよう多めに試行
        v = gen()
        if v not in vals:
            vals.append(v)
        if len(vals) >= n:
            break
    return vals


# negative 用の leak 検出（カテゴリ単位の best-effort 正規表現。完全保証ではない）。
LEAK_PATTERNS: dict[str, re.Pattern] = {
    "network_identifier": re.compile(
        r"(?:\d{1,3}\.){3}\d{1,3}"
        r"|[0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5}"
        r"|[0-9A-Fa-f]{2}(?:-[0-9A-Fa-f]{2}){5}"
        r"|[\w-]+\.(?:local|internal|intranet)\b"
        r"|https?://\S|\\\\\S|smb://\S"
    ),
    "system_config": re.compile(
        r"/(?:var|etc|opt|home|srv|usr)/"
        r"|[A-Za-z]:\\"
        r"|\.(?:log|ya?ml|conf|json|java|py|ini)\b"
        r"|\b\w+\.\w+\b(?=\s*(?:カラム|テーブル))"
        r"|\*/?\d+\s+\*\s+\*\s+\*"
    ),
    "financial_info": re.compile(
        r"[¥$]\s?\d|\d[\d,]*\s*(?:円|万円|億)|\d+(?:\.\d+)?\s*%"
    ),
    "transaction_id": re.compile(
        r"[A-Z]{2,}[-_]?\d{2,}|#\d{3,}|\b\d{3,}-\d{2,}"
    ),
}


# ---------------------------------------------------------------------------
# プロンプト構築（例文のみを書かせる。JSON は書かせない）
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "あなたは日本語のビジネス文書を作成するプロのライターです。"
    "与えられた条件に従って、その場面で実際にやり取りされそうな自然で具体的な日本語の文章を1つだけ作成します。"
    "出力は文章の本文のみとし、JSON・コードブロック・前置き・後書き・"
    "「以下が例文です」等の説明・本文を囲う引用符は一切付けないでください。"
)


def build_positive_prompt(situation: str, subtype_desc: str, values: list[str]) -> str:
    vlist = "\n".join(f"- {v}" for v in values)
    return f"""\
「{situation}」という場面で実際にやり取りされそうな、自然な日本語の文章を1〜3文で作成してください。

次の文字列を、表記を一切変えず**一字一句そのまま**本文中に含めてください（{subtype_desc}）:
{vlist}

注意:
- 上記の文字列以外には、機密情報（人名・メールアドレス・電話番号・住所・実在する企業名・上記以外の識別番号や金額など）を一切含めないでください。
- 日本語で、その場面にふさわしい自然な文体にしてください。
- 本文のみを出力してください（説明・引用符・JSONは不要）。"""


def build_negative_prompt(situation: str, subtype_desc: str, example_formats: list[str]) -> str:
    elist = "\n".join(f"- {e}" for e in example_formats[:4])
    return f"""\
「{situation}」という場面で実際にやり取りされそうな、自然な日本語の文章を1〜3文で作成してください。

この文章は「{subtype_desc}」に関する話題に触れても構いませんが、**具体的な値は一切書かない**ハードネガティブな例文にしてください。
（例:「IPアドレスは確認済みです」「請求書を送付しました」のように、話題には言及しても具体的な番号・パス・金額・アドレス等は書かない）

参考: この文章に**書いてはいけない**具体値の形式例:
{elist}

注意:
- 人名・メールアドレス・電話番号・住所・企業名・金額・各種識別番号・IPアドレス・ファイルパス等、いかなる機密情報も含めないでください。
- 本文のみを出力してください（説明・引用符・JSONは不要）。"""


# ---------------------------------------------------------------------------
# 応答の整形
# ---------------------------------------------------------------------------

def clean_text(content: str) -> str:
    """モデル出力から例文本文を取り出す（フェンス/前後引用符を除去）。"""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    # 全体が引用符で囲まれている場合のみ外す
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()
    if text.startswith("「") and text.endswith("」"):
        text = text[1:-1].strip()
    return text


# ---------------------------------------------------------------------------
# タスク展開
# ---------------------------------------------------------------------------

def build_tasks(breakdown: dict, *, limit_situations: int = 0, reps: int = 1) -> list[dict]:
    """(カテゴリ×サブタイプ×シチュエーション×極性×rep) の全タスクを展開する。"""
    tasks: list[dict] = []
    for category_key in TARGET_CATEGORIES:
        cat = breakdown[category_key]
        category_desc = cat.get("description", "")
        situations = cat.get("situations", [])
        if limit_situations > 0:
            situations = situations[:limit_situations]
        for subtype in cat["subtypes"]:
            sub_name = subtype["name"]
            sub_desc = subtype.get("description", "")
            examples = subtype.get("examples", [])
            for si, situation in enumerate(situations):
                for positive in (True, False):
                    for rep in range(reps):
                        task = {
                            "task_id": f"{category_key}/{sub_name}/{si}/{'pos' if positive else 'neg'}/{rep}",
                            "category_key": category_key,
                            "category_desc": category_desc,
                            "subtype_name": sub_name,
                            "subtype_desc": sub_desc,
                            "examples": examples,
                            "situation": situation,
                            "positive": positive,
                        }
                        if positive:
                            task["entities"] = gen_entities(sub_name)
                        tasks.append(task)
    return tasks


def task_to_messages(task: dict) -> list[Message]:
    if task["positive"]:
        user = build_positive_prompt(task["situation"], task["subtype_desc"], task["entities"])
    else:
        user = build_negative_prompt(task["situation"], task["subtype_desc"], task["examples"])
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# 非同期生成 + 逐次保存
# ---------------------------------------------------------------------------

async def _attempt(task: dict, client: HFInferenceClient, cfg: InferenceConfig,
                   semaphore: asyncio.Semaphore) -> tuple[str, dict[str, list[str]]] | None:
    """1回分の生成 + 検証。成功なら (input_text, annotation)、不適合なら None。"""
    result = await client.async_chat_complete(
        task_to_messages(task), config_override=cfg, semaphore=semaphore
    )
    text = clean_text(result.content)
    if not text:
        return None

    annotation = _empty_annotation()
    if task["positive"]:
        # 生成した全 entity が verbatim で含まれることを要求
        for v in task["entities"]:
            if v not in text:
                return None
        annotation[task["category_key"]] = list(dict.fromkeys(task["entities"]))
    else:
        # ハードネガティブ: 対象カテゴリの値が漏れていないか best-effort で検査
        leak = LEAK_PATTERNS.get(task["category_key"])
        if leak and leak.search(text):
            return None
    return text, annotation


async def generate(tasks: list[dict], client: HFInferenceClient, cfg: InferenceConfig,
                   out_path: Path, *, concurrency: int, max_attempts: int = 2) -> dict[str, int]:
    """各タスクを並列実行し、完了したものから JSONL へ追記する（逐次保存）。"""
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    stats = {"ok": 0, "rejected": 0, "error": 0, "pos": 0, "neg": 0}

    pbar = tqdm(total=len(tasks), desc="synthesizing", dynamic_ncols=True)
    f = out_path.open("a", encoding="utf-8")

    async def worker(task: dict) -> None:
        parsed = None
        try:
            for _ in range(max_attempts):
                parsed = await _attempt(task, client, cfg, semaphore)
                if parsed is not None:
                    break
        except Exception as exc:  # noqa: BLE001 — 1件失敗で全体は止めない
            tqdm.write(f"[error] {task['task_id']}: {exc}")
            async with write_lock:
                stats["error"] += 1
                pbar.update(1)
            return

        async with write_lock:
            if parsed is None:
                stats["rejected"] += 1
            else:
                input_text, annotation = parsed
                row = {
                    "input_text": input_text,
                    "annotation_json": json.dumps(annotation, ensure_ascii=False),
                    "_meta": {
                        "task_id": task["task_id"],
                        "category": task["category_key"],
                        "subtype": task["subtype_name"],
                        "polarity": "pos" if task["positive"] else "neg",
                    },
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                stats["ok"] += 1
                stats["pos" if task["positive"] else "neg"] += 1
            pbar.update(1)

    try:
        await asyncio.gather(*(worker(t) for t in tasks))
    finally:
        f.close()
        pbar.close()
    return stats


# ---------------------------------------------------------------------------
# resume / export ヘルパ
# ---------------------------------------------------------------------------

def load_done_task_ids(out_path: Path) -> set[str]:
    """既存 JSONL から完了済み task_id を読み出す（resume 用）。"""
    done: set[str] = set()
    if not out_path.exists():
        return done
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = (obj.get("_meta") or {}).get("task_id")
            if tid:
                done.add(tid)
    return done


def export_to_arrow(jsonl_path: Path, out_dir: Path) -> None:
    """JSONL を README 準拠の2カラム HF Dataset (Arrow) に変換して保存する。"""
    from datasets import Dataset

    rows: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({
                "input_text": obj["input_text"],
                "annotation_json": obj["annotation_json"],
            })
    ds = Dataset.from_list(rows)
    ds.save_to_disk(str(out_dir))
    print(f"\n  Exported {len(ds):,} rows → {out_dir}")


# ---------------------------------------------------------------------------
# .env 簡易ローダ（python-dotenv 非依存）
# ---------------------------------------------------------------------------

def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("MODEL", "google/gemma-4-26B-A4B-it"),
                        help="生成モデル（既定: google/gemma-4-26B-A4B-it）")
    parser.add_argument("--provider", default=os.environ.get("PROVIDER", "auto"),
                        help="HF Inference Provider（既定: auto。auto はモデルを提供する利用可能なプロバイダを自動選択する。"
                             "deepinfra を明示するには deepinfra 対応版の huggingface_hub が必要）")
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("CONCURRENCY", "16")),
                        help="同時実行数（既定: 16）")
    parser.add_argument("--limit-situations", type=int, default=int(os.environ.get("LIMIT_SITUATIONS", "0")),
                        help="サブタイプごとに使うシチュエーション数の上限（0=全件）")
    parser.add_argument("--reps", type=int, default=int(os.environ.get("REPS", "1")),
                        help="(サブタイプ×シチュエーション×極性) あたりの生成本数（既定: 1）")
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.8")),
                        help="サンプリング温度（多様性確保のため既定 0.8）")
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "640")))
    parser.add_argument("--seed", type=int, default=None, help="entity 値生成の乱数シード（再現用）")
    parser.add_argument("--out", default=None,
                        help="出力 JSONL パス（既定: experiments/data/synthetic/synthetic.jsonl）")
    parser.add_argument("--no-export", action="store_true", help="完了後の Arrow 形式エクスポートを行わない")
    parser.add_argument("--overwrite", action="store_true",
                        help="既存 JSONL を無視して最初から生成し直す（resume しない）")
    parser.add_argument("--no-pool", action="store_true",
                        help="LLM による entity 値プール生成を行わず固定リストを使う（高速テスト用）")
    parser.add_argument("--pool-n", type=int, default=_POOL_N,
                        help=f"プール1サブタイプあたりの生成数（既定: {_POOL_N}）")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    _load_dotenv(REPO_ROOT / ".env")

    here = Path(__file__).resolve().parent
    breakdown = yaml.safe_load((here / "category_breakdown.yaml").read_text(encoding="utf-8"))

    out_path = Path(args.out) if args.out else REPO_ROOT / "experiments/data/synthetic/synthetic.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and out_path.exists():
        out_path.unlink()

    cfg = InferenceConfig(
        model=args.model,
        provider=args.provider,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.95,
    )
    client = HFInferenceClient(cfg)

    # 固定リストでは多様性が低いサブタイプの entity 値を LLM で事前生成する。
    # build_tasks() よりも前に実行することで gen_entities() が LLM プールを使う。
    if not args.no_pool:
        pools = asyncio.run(build_entity_pools(client, cfg, n=args.pool_n))
        patch_entity_generators(pools)

    tasks = build_tasks(breakdown, limit_situations=args.limit_situations, reps=args.reps)

    done = load_done_task_ids(out_path)
    if done:
        before = len(tasks)
        tasks = [t for t in tasks if t["task_id"] not in done]
        print(f"Resume: {len(done):,} 件完了済み → 残り {len(tasks):,} / {before:,} 件を生成")

    if not tasks:
        print("生成対象のタスクがありません（すべて完了済み）。")
    else:
        n_pos = sum(1 for t in tasks if t["positive"])
        print(
            f"Model: {args.model} (provider={args.provider})\n"
            f"Tasks: {len(tasks):,}  (positive={n_pos:,} / negative={len(tasks) - n_pos:,})\n"
            f"Concurrency: {args.concurrency}\n"
            f"Output: {out_path}"
        )

        stats = asyncio.run(generate(tasks, client, cfg, out_path, concurrency=args.concurrency))
        print(
            "\n生成完了:\n"
            f"  成功         : {stats['ok']:,}  (positive={stats['pos']:,} / negative={stats['neg']:,})\n"
            f"  破棄(不適合) : {stats['rejected']:,}\n"
            f"  API エラー   : {stats['error']:,}"
        )

    if not args.no_export and out_path.exists():
        export_dir = REPO_ROOT / "experiments/data/synthetic_processed"
        export_to_arrow(out_path, export_dir)
        print("\n確認:")
        print("  from datasets import load_from_disk")
        print(f"  ds = load_from_disk('{export_dir.relative_to(REPO_ROOT)}')")
        print("  print(ds[0])")


if __name__ == "__main__":
    main()
