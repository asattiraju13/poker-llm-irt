"""Restricted permutation test: shuffle skill labels within (action x phase) strata.

The null preserves the action and phase composition of each skill class, so a
significant result implies skill labels carry signal beyond what action and
phase together explain.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import pandas as pd

from poker_irt.factor import factor_bic
from poker_irt.qmatrix import fit_qmatrix, qmatrix_bic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="main_11k_seed42_7models")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--n-perms", type=int, default=20)
    p.add_argument("--skill-file",
                   default="data/responses/main_11k_seed42_7models/item_skills_HEURISTIC.parquet")
    p.add_argument("--label", default="heuristic")
    args = p.parse_args()

    in_dir = Path(f"outputs/{args.run}/artifacts")
    wide_aa = pd.read_parquet(in_dir / "response_wide_aa.parquet")
    long = pd.read_parquet(in_dir / "response_long.parquet")

    with (in_dir / "factor_fits" / "K1.pkl").open("rb") as f:
        K1 = pickle.load(f)
    bic_K1 = factor_bic(K1, wide_aa)
    print(f"K=1 BIC: {bic_K1:.0f}", flush=True)

    skills = pd.read_parquet(args.skill_file).set_index("item_id")["skill"]
    wide_items = list(wide_aa.index)
    skills = skills.loc[wide_items]

    item_meta = long.drop_duplicates("item_id").set_index("item_id")[["gto_action_class", "split"]]
    item_meta = item_meta.loc[wide_items]

    strata = item_meta["gto_action_class"].astype(str) + "|" + item_meta["split"].astype(str)
    print(f"\nStrata (action × phase) distribution:", flush=True)
    print(strata.value_counts().to_string(), flush=True)

    real_path = in_dir / "qmatrix_fit_skill_heuristic.pkl"
    if not real_path.exists():
        real_path = in_dir / "qmatrix_fit_skill.pkl"
    with real_path.open("rb") as f:
        real_fit = pickle.load(f)
    bic_real = qmatrix_bic(real_fit, wide_aa)
    delta_real = bic_real - bic_K1
    print(f"\nReal {args.label} Q-matrix: LL={real_fit.log_lik:.0f}  BIC={bic_real:.0f}  ΔBIC={delta_real:+.0f}", flush=True)

    unique_dims = sorted(skills.unique())
    d2i = {d: i for i, d in enumerate(unique_dims)}

    print(f"\n=== Restricted permutation test (N={args.n_perms}) ===", flush=True)
    print(f"  Shuffling within (action × phase) strata", flush=True)
    null_deltas = []
    for perm in range(args.n_perms):
        rng = np.random.RandomState(2000 + perm)
        shuffled = skills.copy()
        for stratum_name, idx in strata.groupby(strata).groups.items():
            vals = skills.loc[idx].values.copy()
            rng.shuffle(vals)
            shuffled.loc[idx] = vals
        item_dim = {iid: d2i[shuffled.loc[iid]] for iid in wide_items}
        t0 = time.time()
        fit_p = fit_qmatrix(wide_aa, item_dim, unique_dims, epochs=args.epochs, seed=42)
        bic_p = qmatrix_bic(fit_p, wide_aa)
        delta_p = bic_p - bic_K1
        null_deltas.append(delta_p)
        print(f"  perm {perm+1}/{args.n_perms}: LL={fit_p.log_lik:.0f}  ΔBIC={delta_p:+.0f}  ({time.time()-t0:.1f}s)", flush=True)

    null_deltas = np.array(null_deltas)
    p_val = float(np.mean(null_deltas <= delta_real))
    print(f"\nRestricted null distribution (N={len(null_deltas)}):", flush=True)
    print(f"  mean  = {null_deltas.mean():+.0f}", flush=True)
    print(f"  std   = {null_deltas.std():.0f}", flush=True)
    print(f"  min   = {null_deltas.min():+.0f}  (best null)", flush=True)
    print(f"  max   = {null_deltas.max():+.0f}  (worst null)", flush=True)
    print(f"  q5/q95= {np.quantile(null_deltas, 0.05):+.0f}/{np.quantile(null_deltas, 0.95):+.0f}", flush=True)
    print(f"  real  = {delta_real:+.0f}", flush=True)
    print(f"  signal = {delta_real - null_deltas.mean():+.0f}  ({(delta_real - null_deltas.mean()) / null_deltas.std():.1f} σ)", flush=True)
    print(f"  empirical p = {p_val:.4f}", flush=True)

    pd.DataFrame({"perm": list(range(args.n_perms)),
                  "delta_bic_null": null_deltas}).to_csv(
        in_dir / f"permutation_test_RESTRICTED_{args.label}.csv", index=False)

    summary_path = in_dir / f"permutation_test_RESTRICTED_{args.label}_summary.txt"
    summary_path.write_text(
        f"Restricted permutation test: shuffle within (action × phase) strata\n"
        f"Label set: {args.label}\n"
        f"Run: {args.run}\n"
        f"N_perm: {args.n_perms}\n\n"
        f"Real ΔBIC ({args.label} Q-matrix - K=1): {delta_real:+.0f}\n"
        f"Restricted null mean ΔBIC: {null_deltas.mean():+.0f} ± {null_deltas.std():.0f}\n"
        f"Restricted null min:        {null_deltas.min():+.0f}\n"
        f"Restricted null max:        {null_deltas.max():+.0f}\n\n"
        f"Effect size (real - null mean): {delta_real - null_deltas.mean():+.0f} BIC\n"
        f"In std units: {(delta_real - null_deltas.mean()) / null_deltas.std():.1f} σ\n"
        f"Empirical p-value: {p_val:.4f}\n"
    )
    print(f"\nsaved {summary_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
