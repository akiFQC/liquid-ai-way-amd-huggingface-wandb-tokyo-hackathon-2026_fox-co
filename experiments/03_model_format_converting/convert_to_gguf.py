# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch>=2.8,<2.13",
#   "transformers>=4.45",
#   "huggingface_hub>=0.25",
#   "sentencepiece",
#   "protobuf",
#   "gguf>=0.9",
#   "numpy>=1.20",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu126"
# url = "https://download.pytorch.org/whl/cu126"
# explicit = true
#
# [tool.uv.sources]
# torch = [
#   { index = "pytorch-cu126", marker = "platform_system == 'Linux'" },
# ]
# ///
"""HF モデル → GGUF 変換 + HF Hub アップロードスクリプト。

llama.cpp を実行時にクローン・ビルドし、convert_hf_to_gguf.py と
llama-quantize を使って指定の量子化バリアントを生成する。

Env overrides (all optional unless noted):

    SOURCE_MODEL        変換元 HF モデル repo id
                        (default: akiFQC/LFM2.5-1.2B-JP-202606-Conf-Extract)
    TARGET_REPO         アップロード先 HF repo id
                        (default: akiFQC/LFM2.5-1.2B-JP-202606-Conf-Extract-GGUF)
    QUANT_TYPES         カンマ区切りの量子化タイプ一覧 (default: Q4_K_M,Q8_0,BF16)
                        llama-quantize に渡す文字列をそのまま使用:
                          Q4_K_M = Int4, Q8_0 = Int8, BF16 = BFloat16
    CREATE_PUSH_TARGET  1 = ターゲット repo を自動作成 (default: 1)
    WORK_DIR            一時作業ディレクトリ (default: /tmp/gguf-convert)
    HF_TOKEN            必須。write スコープが必要。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SOURCE_MODEL = os.environ.get(
    "SOURCE_MODEL", "akiFQC/LFM2.5-1.2B-JP-202606-Conf-Extract"
)
TARGET_REPO = os.environ.get(
    "TARGET_REPO", "akiFQC/LFM2.5-1.2B-JP-202606-Conf-Extract-GGUF"
)
_quant_types_raw = os.environ.get("QUANT_TYPES", "Q4_K_M,Q8_0,BF16")
QUANT_TYPES: list[str] = [q.strip() for q in _quant_types_raw.split(",") if q.strip()]
CREATE_PUSH_TARGET = os.environ.get("CREATE_PUSH_TARGET", "1") in ("1", "true", "True", "TRUE")
WORK_DIR = Path(os.environ.get("WORK_DIR", "/tmp/gguf-convert"))

LLAMA_CPP_DIR = WORK_DIR / "llama.cpp"
MODEL_DIR = WORK_DIR / "hf_model"
OUTPUT_DIR = WORK_DIR / "output"


def _run(cmd: list, cwd: Path | None = None) -> None:
    print(f"[run] {' '.join(str(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd, check=False, cwd=cwd)
    if result.returncode != 0:
        sys.exit(f"[convert] command failed (exit {result.returncode}): {cmd[0]}")


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        sys.exit("[convert] HF_TOKEN not set; populate .env or export it")

    print(f"[convert] source:      {SOURCE_MODEL}")
    print(f"[convert] target:      {TARGET_REPO}")
    print(f"[convert] quant types: {QUANT_TYPES}")
    print(f"[convert] work dir:    {WORK_DIR}", flush=True)

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=hf_token)

    # Step 1: ターゲット repo を作成（存在しなければ）
    if CREATE_PUSH_TARGET:
        print(f"\n[convert] ensuring target repo exists: {TARGET_REPO}", flush=True)
        api.create_repo(TARGET_REPO, repo_type="model", private=False, exist_ok=True)

    # Step 2: ソースモデルをダウンロード
    print(f"\n[convert] downloading {SOURCE_MODEL} -> {MODEL_DIR}", flush=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=SOURCE_MODEL,
        local_dir=str(MODEL_DIR),
        token=hf_token,
        ignore_patterns=["*.gguf"],
    )

    # Step 3: llama.cpp をクローン
    print(f"\n[convert] cloning llama.cpp (shallow) -> {LLAMA_CPP_DIR}", flush=True)
    if LLAMA_CPP_DIR.exists():
        print("[convert] llama.cpp already cloned, skipping")
    else:
        LLAMA_CPP_DIR.parent.mkdir(parents=True, exist_ok=True)
        _run([
            "git", "clone", "--depth", "1",
            "https://github.com/ggerganov/llama.cpp",
            str(LLAMA_CPP_DIR),
        ])

    # Step 4: llama.cpp の Python 依存関係は UV script ヘッダーで宣言済み。
    # (gguf, torch, transformers, sentencepiece, protobuf, numpy)
    # pip が UV 仮想環境内に存在しないため requirements.txt のインストールは不要。
    # convert_hf_to_gguf.py は sys.path.insert(0, llama.cpp/) で llama.cpp 内の
    # gguf パッケージを優先するため、PyPI 版の gguf と競合しない。

    # Step 4b: cmake / build-essential が未インストールなら apt-get で取得。
    # HF Jobs コンテナは Ubuntu ベースで root 権限があるが cmake はデフォルト未収録。
    import shutil
    missing_tools = [t for t in ("cmake", "gcc", "g++") if not shutil.which(t)]
    if missing_tools:
        print(f"\n[convert] missing build tools {missing_tools}, installing via apt-get...", flush=True)
        _run(["apt-get", "update", "-qq"])
        _run(["apt-get", "install", "-y", "-qq", "cmake", "build-essential"])

    # Step 5: llama-quantize バイナリをビルド
    build_dir = LLAMA_CPP_DIR / "build"
    print("\n[convert] cmake configure...", flush=True)
    _run([
        "cmake", "-B", str(build_dir),
        "-DGGML_VULKAN=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
        str(LLAMA_CPP_DIR),
    ])
    print("\n[convert] cmake build llama-quantize...", flush=True)
    _run([
        "cmake", "--build", str(build_dir),
        "--target", "llama-quantize",
        "-j", str(os.cpu_count() or 4),
    ])
    quantize_bin = build_dir / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = build_dir / "llama-quantize"
    if not quantize_bin.exists():
        sys.exit(f"[convert] llama-quantize binary not found under {build_dir}")
    print(f"[convert] llama-quantize: {quantize_bin}", flush=True)

    # Step 6: HF → F16 GGUF 変換
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_slug = SOURCE_MODEL.split("/")[-1]
    f16_gguf = OUTPUT_DIR / f"{model_slug}-F16.gguf"

    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    print(f"\n[convert] HF -> F16 GGUF: {f16_gguf}", flush=True)
    _run([
        sys.executable,
        str(convert_script),
        str(MODEL_DIR),
        "--outfile", str(f16_gguf),
        "--outtype", "f16",
    ])

    # Step 7: 各量子化バリアントを生成
    quant_files: list[Path] = []
    for quant_type in QUANT_TYPES:
        quant_gguf = OUTPUT_DIR / f"{model_slug}-{quant_type}.gguf"
        print(f"\n[convert] quantize F16 -> {quant_type}: {quant_gguf}", flush=True)
        _run([str(quantize_bin), str(f16_gguf), str(quant_gguf), quant_type])
        quant_files.append(quant_gguf)

    # Step 8: F16 + 全量子化バリアントをアップロード
    upload_files = [f16_gguf, *quant_files]
    print(f"\n[convert] uploading {len(upload_files)} file(s) to {TARGET_REPO}", flush=True)
    for gguf_file in upload_files:
        if not gguf_file.exists():
            print(f"[convert] WARNING: {gguf_file} not found, skipping")
            continue
        size_gb = gguf_file.stat().st_size / 1e9
        print(f"[convert] uploading {gguf_file.name} ({size_gb:.2f} GB)...", flush=True)
        api.upload_file(
            path_or_fileobj=str(gguf_file),
            path_in_repo=gguf_file.name,
            repo_id=TARGET_REPO,
            repo_type="model",
            token=hf_token,
            commit_message=f"Upload {gguf_file.name}",
        )
        print(f"[convert]   -> uploaded: {gguf_file.name}")

    # Step 9: モデルカードを生成してアップロード
    quant_table_rows = "\n".join(
        f"| `{f.name}` | {qt} |"
        for f, qt in zip(quant_files, QUANT_TYPES)
    )
    llama_cli_examples = "\n\n".join(
        f"```bash\n# {qt}\n./llama-cli \\\n"
        f"  --model ./{f.name} \\\n"
        f"  --n-gpu-layers 99 \\\n"
        f"  --temp 0.3 --min-p 0.15 --repeat-penalty 1.05 \\\n"
        f"  --chat-template lfm2 \\\n"
        f"  --interactive --color\n```"
        for f, qt in zip(quant_files, QUANT_TYPES)
    )
    readme_content = f"""---
base_model: {SOURCE_MODEL}
tags:
  - gguf
  - llama-cpp
---

# {TARGET_REPO.split("/")[-1]}

GGUF conversion of [{SOURCE_MODEL}](https://huggingface.co/{SOURCE_MODEL}).

## Files

| File | Quantization |
|------|-------------|
| `{f16_gguf.name}` | F16 (full precision, for re-quantization) |
{quant_table_rows}

## Usage with llama.cpp

{llama_cli_examples}
"""
    readme_path = OUTPUT_DIR / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"\n[convert] uploading model card...", flush=True)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=TARGET_REPO,
        repo_type="model",
        token=hf_token,
        commit_message="Add model card",
    )

    print(f"\n[convert] Done.")
    print(f"[convert] Model: https://huggingface.co/{TARGET_REPO}")


if __name__ == "__main__":
    main()
