#!/usr/bin/env python3
"""Local dataset preflight, run by the text launcher before submit:
stream the first rows of DATASET through the same `_to_messages`
schema check the training job runs, so typos and column mismatches
surface in ~5 s locally instead of after a ~5 min submit → bootstrap →
fail round trip. Exit 0 = rows pass (or DATASET unset); 1 = failure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PROBE_ROWS = 3  # bounded streaming budget; ~5s


def main() -> int:
    dataset = os.environ.get("DATASET", "").strip()
    if not dataset:
        # No DATASET override → the text launcher uses the default
        # (FineTome), validated upstream by the kit's tests.
        return 0
    config = os.environ.get("DATASET_CONFIG", "").strip() or None
    split = os.environ.get("DATASET_SPLIT", "train").strip() or "train"

    print(f"\nProbing DATASET={dataset!r} (config={config!r}, split={split!r})...")
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from datasets import load_dataset
    from text.train import _parse_dataset_mapper, _to_messages

    mapper = _parse_dataset_mapper(os.environ.get("DATASET_MAPPER"))
    try:
        ds = load_dataset(dataset, config, split=split, streaming=True)
    except Exception as e:
        print(
            f"  ✗ load_dataset({dataset!r}, config={config!r}, split={split!r}) "
            f"failed ({type(e).__name__}: {e})."
        )
        return 1
    for i, row in enumerate(ds):
        if i >= _PROBE_ROWS:
            break
        try:
            _to_messages(row, mapper)
        except Exception as e:
            print(
                f"  ✗ row {i} of {dataset!r} failed schema check "
                f"({type(e).__name__}: {e}). "
                f"For non-standard column layouts, set "
                f"DATASET_MAPPER='user=col_a,assistant=col_b'."
            )
            return 1
    print(f"  ✓ DATASET={dataset!r} probe: {_PROBE_ROWS} rows pass text schema check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
