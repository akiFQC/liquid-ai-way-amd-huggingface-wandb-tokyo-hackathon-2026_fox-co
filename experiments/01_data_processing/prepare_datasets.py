import json
from pathlib import Path

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Label mapping tables (from README カテゴリ対応表)
# ---------------------------------------------------------------------------

OPENPII_LABEL_MAP: dict[str, str] = {
    "GIVENNAME": "human_name",
    "SURNAME": "human_name",
    "CITY": "address",
    "STREETBUILDINGNUM": "address",
    "ZIPCODE": "address",
    "EMAIL": "email_address",
    "TELEPHONENUM": "phone_number",
    "TITLE": "company_name",
    "USERNAME": "account_identifier",
    "IDCARDNUM": "account_identifier",
    "DRIVERLICENSENUM": "account_identifier",
    "SOCIALNUM": "account_identifier",
    "CREDITCARDNUMBER": "financial_info",
    "TAXNUM": "financial_info",
}

WIKIPEDIA_NER_LABEL_MAP: dict[str, str] = {
    "人名": "human_name",
    "法人名": "company_name",
    "政治的組織名": "company_name",
    "その他の組織名": "company_name",
    "地名": "address",
    "施設名": "address",
    "製品名": "project_info",
}

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


def _empty_annotation() -> dict[str, list]:
    return {k: [] for k in ALL_KEYS}


def _print_stats(ds, label: str) -> None:
    total = len(ds)
    print(f"\n  [{label}] {total:,} rows")
    print(f"  {'カテゴリ':<30}  {'件数':>8}  {'行数':>8}  {'行カバー率':>10}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*10}")
    for key in ALL_KEYS:
        entity_counts = [len(json.loads(row["annotation_json"])[key]) for row in ds]
        total_entities = sum(entity_counts)
        rows_with_entity = sum(1 for c in entity_counts if c > 0)
        coverage = rows_with_entity / total * 100 if total > 0 else 0.0
        print(f"  {key:<30}  {total_entities:>8,}  {rows_with_entity:>8,}  {coverage:>9.1f}%")


# ---------------------------------------------------------------------------
# OpenPII 1.5M  (ai4privacy/pii-masking-openpii-1.5m)
# Expected columns: source_text, privacy_mask (list of {value, label, start, end})
# ---------------------------------------------------------------------------

def _process_openpii_row(row: dict) -> dict:
    annotation = _empty_annotation()
    for entity in row["privacy_mask"]:
        key = OPENPII_LABEL_MAP.get(entity["label"])
        if key:
            annotation[key].append(entity["value"])
    return {
        "input_text": row["source_text"],
        "annotation_json": json.dumps(annotation, ensure_ascii=False),
    }


def process_openpii(data_dir: Path) -> None:
    print("Loading ai4privacy/pii-masking-openpii-1.5m …")
    ds = load_dataset("ai4privacy/pii-masking-openpii-1.5m", split="train")
    ja = ds.filter(
        lambda row: row["language"] == "ja",
        desc="Filtering Japanese rows",
    )
    print(f"  Japanese subset: {len(ja):,} / {len(ds):,} rows")
    processed = ja.map(
        _process_openpii_row,
        remove_columns=ja.column_names,
        desc="OpenPII → target schema",
    )
    _print_stats(processed, "OpenPII 1.5M (ja)")
    out = data_dir / "openpii_processed"
    processed.save_to_disk(str(out))
    print(f"\n  Saved {len(processed):,} rows → {out}")


# ---------------------------------------------------------------------------
# ner-wikipedia-dataset  (stockmarkteam/ner-wikipedia-dataset)
# Expected columns: text, entities (list of {name, type, span_start, span_end})
# ---------------------------------------------------------------------------

def _process_wikipedia_row(row: dict) -> dict:
    annotation = _empty_annotation()
    for entity in row["entities"]:
        key = WIKIPEDIA_NER_LABEL_MAP.get(entity["type"])
        if key:
            annotation[key].append(entity["name"])
    return {
        "input_text": row["text"],
        "annotation_json": json.dumps(annotation, ensure_ascii=False),
    }


def process_wikipedia(data_dir: Path) -> None:
    print("Loading stockmarkteam/ner-wikipedia-dataset …")
    ds = load_dataset("stockmark/ner-wikipedia-dataset", split="train")
    processed = ds.map(
        _process_wikipedia_row,
        remove_columns=ds.column_names,
        desc="ner-wikipedia → target schema",
    )
    _print_stats(processed, "ner-wikipedia-dataset")
    out = data_dir / "ner_wikipedia_processed"
    processed.save_to_disk(str(out))
    print(f"\n  Saved {len(processed):,} rows → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    process_openpii(data_dir)
    process_wikipedia(data_dir)

    print("\nDone. Verify with:")
    print("  from datasets import load_from_disk")
    print("  ds = load_from_disk('experiments/data/openpii_processed')")
    print("  print(ds[0])")


if __name__ == "__main__":
    main()
