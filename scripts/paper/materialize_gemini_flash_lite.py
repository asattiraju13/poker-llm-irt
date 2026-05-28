"""Materialize the Gemini Flash-Lite JSONL log into per-split parquet shards."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

DEST_DIR = Path("data/responses/main_11k_seed42_7models")
SRC_JSONL = DEST_DIR / "_logs" / "google__gemini-2.5-flash-lite.jsonl"
SAFE_NAME = "google__gemini-2.5-flash-lite"


def main() -> int:
    rows = []
    with SRC_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        print("no rows", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    df = df.sort_values("ts").drop_duplicates("item_id", keep="last")
    print(f"loaded {len(df)} unique items from {SRC_JSONL}")
    print(f"per-split counts: {df.groupby('split').size().to_dict()}")
    print(f"AA: {df['aa'].mean():.4f}  EM: {df['em'].mean():.4f}  parse_ok: {df['parse_ok'].mean():.4f}")
    print(f"error rate: {(df['error'] != '').mean():.4f}")

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for split, sub in df.groupby("split"):
        out = DEST_DIR / f"{SAFE_NAME}__{split}.parquet"
        sub.to_parquet(out, index=False)
        print(f"  wrote {out}  ({len(sub)} rows)")

    log_dest = DEST_DIR / "_logs" / f"{SAFE_NAME}.jsonl"
    log_dest.parent.mkdir(parents=True, exist_ok=True)
    log_dest.write_text(SRC_JSONL.read_text())
    print(f"  copied jsonl to {log_dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
