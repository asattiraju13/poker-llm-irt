"""Build a heuristic skill Q-matrix from item features and the GTO action label.

Skill taxonomy (rule-based, first match wins):
``preflop_open_3bet_4bet``, ``folding_discipline``, ``value_betting``,
``bluff_betting``, ``pot_odds_drawing``, ``bluff_catching``,
``pot_control_check``, ``position_aware_continuation``.

Output: parquet with columns ``item_id, skill``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def label_item(row) -> str:
    phase = row["game_phase"]
    if phase == "preflop":
        return "preflop_open_3bet_4bet"

    gto = row["gto_action_class"]
    hs = int(row.get("hand_strength_class") or 0)
    flush_draw = bool(row.get("hero_has_flush_draw") or 0)
    straight_draw = bool(row.get("hero_straight_draw") or 0)
    has_draw = flush_draw or straight_draw

    if gto == "fold":
        return "folding_discipline"
    if gto == "bet":
        if hs >= 2:
            return "value_betting"
        if hs == 0:
            return "bluff_betting"
        return "bluff_betting" if has_draw else "value_betting"
    if gto == "call":
        if has_draw and hs <= 1:
            return "pot_odds_drawing"
        return "bluff_catching"
    if gto == "check":
        if hs >= 1:
            return "pot_control_check"
        return "position_aware_continuation"
    return "value_betting"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--features",
                   default="outputs/main_11k_seed42_7models/artifacts/item_features_v3.parquet",
                   help="path to item_features_v3.parquet")
    p.add_argument("--long",
                   default="outputs/main_11k_seed42_7models/artifacts/response_long.parquet",
                   help="path to response_long.parquet (for gto_action_class)")
    p.add_argument("--out",
                   default="data/responses/main_11k_seed42_7models/item_skills_HEURISTIC.parquet",
                   help="output parquet path")
    args = p.parse_args()

    feats = pd.read_parquet(args.features)
    long = pd.read_parquet(args.long).drop_duplicates("item_id")[["item_id", "gto_action_class"]]
    df = feats.merge(long, on="item_id", how="left")

    df["skill"] = df.apply(label_item, axis=1)

    out = df[["item_id", "skill"]].copy()
    print(f"Total items: {len(out)}")
    print(f"Skill distribution:\n{out['skill'].value_counts().to_string()}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"saved → {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
