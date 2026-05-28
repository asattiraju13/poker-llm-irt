"""5-fold cell-held-out log-likelihood for Q-matrix variants vs the K=1 baseline."""
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

from poker_irt.factor import fit_factor_model, _wide_to_tensor
from poker_irt.qmatrix import fit_qmatrix


def held_out_ll_qmatrix(wide_aa, item_dim_map, dim_names, n_folds=5, epochs=3000, seed=42):
    """K-fold CV over response cells for the Q-matrix model; returns mean LL per cell."""
    Y_full, M_full = _wide_to_tensor(wide_aa)
    n_subj, n_item = Y_full.shape
    rng = np.random.RandomState(seed)

    obs_idx = np.argwhere(M_full.numpy() > 0)
    rng.shuffle(obs_idx)
    fold_size = len(obs_idx) // n_folds

    total_ll = 0.0
    total_n = 0
    for f in range(n_folds):
        test_indices = obs_idx[f * fold_size: (f + 1) * fold_size]
        wide_train = wide_aa.copy()
        for i, j in test_indices:
            wide_train.iloc[j, i] = np.nan
        fit = fit_qmatrix(wide_train, item_dim_map, dim_names,
                          epochs=epochs, seed=seed + f)
        U64 = np.nan_to_num(fit.U.astype(np.float64), nan=0.0, posinf=50.0, neginf=-50.0)
        V64 = np.nan_to_num(fit.V.astype(np.float64), nan=0.0, posinf=50.0, neginf=-50.0)
        z64 = np.nan_to_num(fit.z.astype(np.float64), nan=0.0, posinf=50.0, neginf=-50.0)
        item_dim = np.array([item_dim_map[wide_aa.index[j]] for j in range(n_item)])
        U_per_item = U64[:, item_dim]
        logits = U_per_item * V64[None, :] - z64[None, :]
        logits = np.clip(logits, -50.0, 50.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        p = np.clip(p, 1e-8, 1.0 - 1e-8)
        Yn = Y_full.numpy().astype(np.float64)
        for i, j in test_indices:
            y = Yn[i, j]
            total_ll += y * np.log(p[i, j]) + (1 - y) * np.log(1 - p[i, j])
        total_n += len(test_indices)
        print(f"  fold {f+1}/{n_folds}: test_n={len(test_indices)}", flush=True)

    return float(total_ll / max(total_n, 1)), int(total_n)


def held_out_ll_free(wide_aa, K, n_folds=5, epochs=3000, seed=42):
    """Same procedure for the free K-factor model."""
    from poker_irt.factor import held_out_log_lik
    return held_out_log_lik(wide_aa, K=K, n_folds=n_folds, epochs=epochs, seed=seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="main_11k_seed42_7models")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--gemini-skill-file",
                   default="data/responses/main_11k_seed42_7models/item_skills_at_11k.parquet")
    p.add_argument("--heuristic-skill-file",
                   default="data/responses/main_11k_seed42_7models/item_skills_HEURISTIC.parquet")
    args = p.parse_args()

    in_dir = Path(f"outputs/{args.run}/artifacts")
    wide_aa = pd.read_parquet(in_dir / "response_wide_aa.parquet")
    long = pd.read_parquet(in_dir / "response_long.parquet")
    n_subj = wide_aa.shape[1]
    n_item = wide_aa.shape[0]
    print(f"Loaded {n_item} items × {n_subj} models from {args.run}", flush=True)

    results = []

    # K=1 free factor baseline
    print("\n=== K=1 free factor (baseline) ===", flush=True)
    t0 = time.time()
    cv_ll = held_out_ll_free(wide_aa, K=1, n_folds=args.n_folds, epochs=args.epochs, seed=42)
    print(f"K=1 CV LL/cell = {cv_ll:.4f}  ({time.time()-t0:.1f}s)", flush=True)
    results.append({"variant": "K=1 free factor", "K": 1, "cv_ll_per_cell": cv_ll,
                    "wall_s": time.time()-t0})

    # Action GTO Q-matrix
    print("\n=== Action GTO Q-matrix (K=4) ===", flush=True)
    item_dim_raw = long.drop_duplicates("item_id").set_index("item_id")["gto_action_class"]
    dims_action = sorted(item_dim_raw.unique())
    d2i = {d: i for i, d in enumerate(dims_action)}
    item_dim_action = {iid: d2i[d] for iid, d in item_dim_raw.items()}
    t0 = time.time()
    cv_ll, n_held = held_out_ll_qmatrix(wide_aa, item_dim_action, dims_action,
                                        n_folds=args.n_folds, epochs=args.epochs, seed=42)
    print(f"Action CV LL/cell = {cv_ll:.4f}  ({time.time()-t0:.1f}s)", flush=True)
    results.append({"variant": "Q-matrix Action GTO", "K": 4, "cv_ll_per_cell": cv_ll,
                    "wall_s": time.time()-t0})

    # Phase Q-matrix
    print("\n=== Phase Q-matrix (K=4) ===", flush=True)
    item_dim_raw = long.drop_duplicates("item_id").set_index("item_id")["split"]
    dims_phase = sorted(item_dim_raw.unique())
    d2i = {d: i for i, d in enumerate(dims_phase)}
    item_dim_phase = {iid: d2i[d] for iid, d in item_dim_raw.items()}
    t0 = time.time()
    cv_ll, _ = held_out_ll_qmatrix(wide_aa, item_dim_phase, dims_phase,
                                   n_folds=args.n_folds, epochs=args.epochs, seed=42)
    print(f"Phase CV LL/cell = {cv_ll:.4f}  ({time.time()-t0:.1f}s)", flush=True)
    results.append({"variant": "Q-matrix Phase", "K": len(dims_phase), "cv_ll_per_cell": cv_ll,
                    "wall_s": time.time()-t0})

    # Heuristic skill Q-matrix
    print("\n=== Heuristic skill Q-matrix (K=8) ===", flush=True)
    skills_h = pd.read_parquet(args.heuristic_skill_file).set_index("item_id")["skill"]
    wide_items = set(wide_aa.index)
    skills_h = skills_h[skills_h.index.isin(wide_items)]
    dims_h = sorted(skills_h.unique())
    d2i = {d: i for i, d in enumerate(dims_h)}
    item_dim_h = {iid: d2i[d] for iid, d in skills_h.items()}
    t0 = time.time()
    cv_ll, _ = held_out_ll_qmatrix(wide_aa, item_dim_h, dims_h,
                                   n_folds=args.n_folds, epochs=args.epochs, seed=42)
    print(f"Heuristic CV LL/cell = {cv_ll:.4f}  ({time.time()-t0:.1f}s)", flush=True)
    results.append({"variant": "Q-matrix Skill Heuristic", "K": len(dims_h),
                    "cv_ll_per_cell": cv_ll, "wall_s": time.time()-t0})

    # Gemini skill Q-matrix
    print("\n=== Gemini skill Q-matrix (K=8) ===", flush=True)
    skills_g = pd.read_parquet(args.gemini_skill_file).set_index("item_id")["skill"]
    skills_g = skills_g[skills_g.index.isin(wide_items)]
    dims_g = sorted(skills_g.unique())
    d2i = {d: i for i, d in enumerate(dims_g)}
    item_dim_g = {iid: d2i[d] for iid, d in skills_g.items()}
    t0 = time.time()
    cv_ll, _ = held_out_ll_qmatrix(wide_aa, item_dim_g, dims_g,
                                   n_folds=args.n_folds, epochs=args.epochs, seed=42)
    print(f"Gemini CV LL/cell = {cv_ll:.4f}  ({time.time()-t0:.1f}s)", flush=True)
    results.append({"variant": "Q-matrix Skill Gemini", "K": len(dims_g),
                    "cv_ll_per_cell": cv_ll, "wall_s": time.time()-t0})

    df = pd.DataFrame(results)
    df = df.sort_values("cv_ll_per_cell", ascending=False).reset_index(drop=True)
    baseline = df[df["variant"] == "K=1 free factor"]["cv_ll_per_cell"].iloc[0]
    df["delta_vs_K1"] = df["cv_ll_per_cell"] - baseline
    out_csv = in_dir / "qmatrix_cv_ll.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}\n", flush=True)
    print(df.to_string(index=False), flush=True)

    rep = Path(f"outputs/{args.run}/QMATRIX_CV_LL.md")
    rep.write_text(
        "# 5-fold cross-validated held-out log-likelihood\n\n"
        "Per-cell mean held-out log-likelihood (higher is better).\n\n"
        + df.to_markdown(index=False) + "\n\n"
        f"Total observed cells: {n_subj * n_item:,}.  Held-out per fold: ~{(n_subj * n_item)//5:,}.\n"
    )
    print(f"saved {rep}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
