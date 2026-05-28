"""Unrestricted permutation test for the skill-Q-matrix BIC delta.

Generates ``--n-perms`` random label permutations (preserving the marginal
class distribution), refits the Q-matrix on each, and reports the null
distribution. Run ``qmatrix_comparison.py --dimension skill`` first so the
real fit is on disk.
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--n-perms", type=int, default=20)
    p.add_argument("--skill-file", required=True)
    args = p.parse_args()

    in_dir = Path(f"outputs/{args.run}/artifacts")
    wide_aa = pd.read_parquet(in_dir / "response_wide_aa.parquet")
    skills = pd.read_parquet(args.skill_file).set_index("item_id")["skill"]

    with (in_dir / "factor_fits" / "K1.pkl").open("rb") as f:
        fit_uncon = pickle.load(f)
    bic_uncon = factor_bic(fit_uncon, wide_aa)
    print(f"K=1 free factor: LL={fit_uncon.log_lik:.2f}  BIC={bic_uncon:.2f}")

    real_fit_path = in_dir / "qmatrix_fit_skill.pkl"
    if real_fit_path.exists():
        with real_fit_path.open("rb") as f:
            real_fit = pickle.load(f)
        bic_real = qmatrix_bic(real_fit, wide_aa)
        delta_real = bic_real - bic_uncon
        print(f"Real skill Q-matrix: LL={real_fit.log_lik:.2f}  BIC={bic_real:.2f}  ΔBIC={delta_real:+.2f}")
    else:
        raise SystemExit("Real Q-matrix fit not found — run qmatrix_comparison.py first")

    wide_items = set(wide_aa.index)
    skills_real = skills[skills.index.isin(wide_items)].copy()
    unique_dims = sorted(skills_real.unique())
    dim_to_int = {d: i for i, d in enumerate(unique_dims)}

    null_deltas = []
    for perm in range(args.n_perms):
        rng = np.random.RandomState(1000 + perm)
        shuffled_labels = rng.permutation(skills_real.values)
        item_dim_map = {iid: dim_to_int[d] for iid, d in zip(skills_real.index, shuffled_labels)}

        t0 = time.time()
        fit_p = fit_qmatrix(wide_aa, item_dim_map, unique_dims, epochs=args.epochs, seed=42)
        bic_p = qmatrix_bic(fit_p, wide_aa)
        delta_p = bic_p - bic_uncon
        null_deltas.append(delta_p)
        print(f"  perm {perm+1}/{args.n_perms}: LL={fit_p.log_lik:.0f}  ΔBIC={delta_p:+.0f}  ({time.time()-t0:.1f}s)")

    null_deltas = np.array(null_deltas)
    print(f"\nNull distribution of ΔBIC (N={len(null_deltas)}):")
    print(f"  mean  = {null_deltas.mean():+.0f}")
    print(f"  std   = {null_deltas.std():.0f}")
    print(f"  min   = {null_deltas.min():+.0f}  (best null)")
    print(f"  max   = {null_deltas.max():+.0f}  (worst null)")
    print(f"  q5    = {np.quantile(null_deltas, 0.05):+.0f}")
    print(f"  q95   = {np.quantile(null_deltas, 0.95):+.0f}")
    print(f"\nReal ΔBIC = {delta_real:+.0f}")
    print(f"Effect size vs null: real - null_mean = {delta_real - null_deltas.mean():+.0f}")
    print(f"Real beats best null by: {delta_real - null_deltas.min():+.0f}")
    p_val = float(np.mean(null_deltas <= delta_real))
    print(f"Empirical p-value: {p_val:.4f}  (fraction of nulls more extreme than real)")

    out_path = in_dir / "permutation_test_skill.csv"
    pd.DataFrame({
        "perm": list(range(args.n_perms)),
        "delta_bic_null": null_deltas,
    }).to_csv(out_path, index=False)
    print(f"saved → {out_path}")

    summary_path = in_dir / "permutation_test_skill_summary.txt"
    summary_path.write_text(
        f"Permutation test on skill Q-matrix labels (N={len(null_deltas)} shuffles)\n"
        f"Run: {args.run}\n"
        f"Real ΔBIC (Q-matrix - K=1): {delta_real:+.0f}\n"
        f"Null mean ΔBIC: {null_deltas.mean():+.0f} ± {null_deltas.std():.0f}\n"
        f"Null min ΔBIC: {null_deltas.min():+.0f}\n"
        f"Null max ΔBIC: {null_deltas.max():+.0f}\n"
        f"Real beats best null by: {delta_real - null_deltas.min():+.0f} BIC points\n"
        f"Empirical p-value: {p_val:.4f}\n"
    )
    print(f"saved → {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
