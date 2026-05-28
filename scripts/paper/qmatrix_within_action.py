"""Joint Action x Skill Q-matrix: test whether skill labels add signal beyond action class."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import pandas as pd

from poker_irt.factor import fit_factor_model, factor_bic, _wide_to_tensor
from poker_irt.qmatrix import fit_qmatrix, qmatrix_bic


def held_out_ll_qmatrix(wide_aa, item_dim_map, dim_names, n_folds=5, epochs=3000, seed=42):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="main_11k_seed42_7models")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--heuristic-skill-file",
                   default="data/responses/main_11k_seed42_7models/item_skills_HEURISTIC.parquet")
    args = p.parse_args()

    in_dir = Path(f"outputs/{args.run}/artifacts")
    wide_aa = pd.read_parquet(in_dir / "response_wide_aa.parquet")
    long = pd.read_parquet(in_dir / "response_long.parquet")
    n_subj = wide_aa.shape[1]; n_item = wide_aa.shape[0]
    print(f"Loaded {n_item} items × {n_subj} models", flush=True)

    item_act = long.drop_duplicates("item_id").set_index("item_id")["gto_action_class"]
    skills_h = pd.read_parquet(args.heuristic_skill_file).set_index("item_id")["skill"]
    wide_items = list(wide_aa.index)
    item_act = item_act.loc[wide_items]
    skills_h = skills_h.loc[wide_items]

    joint_label = item_act.astype(str) + ":" + skills_h.astype(str)
    joint_counts = joint_label.value_counts()
    print(f"\nJoint Action × Skill cells (non-empty): {len(joint_counts)}", flush=True)
    print(joint_counts.head(25).to_string(), flush=True)
    print(f"\nTotal joint cells: {len(joint_counts)}", flush=True)

    unique_joint = sorted(joint_label.unique())
    j2i = {d: i for i, d in enumerate(unique_joint)}
    item_dim_joint = {iid: j2i[joint_label.loc[iid]] for iid in wide_items}

    print(f"\n=== Joint Action × Skill Q-matrix (K={len(unique_joint)}) ===", flush=True)
    best = None
    for s in range(args.n_seeds):
        t0 = time.time()
        fit = fit_qmatrix(wide_aa, item_dim_joint, unique_joint,
                          epochs=args.epochs, seed=42 + s)
        print(f"  seed {42+s}: LL={fit.log_lik:.0f}  ({time.time()-t0:.1f}s)", flush=True)
        if best is None or fit.log_lik > best.log_lik:
            best = fit
    bic_joint = qmatrix_bic(best, wide_aa)
    print(f"Joint Best: LL={best.log_lik:.0f}  BIC={bic_joint:.0f}", flush=True)

    import pickle
    with open(in_dir / "qmatrix_fit_joint_action_skill.pkl", "wb") as f:
        pickle.dump(best, f)
    prof = pd.DataFrame(best.U, index=best.subject_order, columns=unique_joint)
    prof.to_csv(in_dir / "qmatrix_profile_joint_action_skill.csv")
    print(f"saved fit + profile", flush=True)

    print(f"\n=== CV LL for joint Q-matrix (5-fold) ===", flush=True)
    t0 = time.time()
    cv_ll_joint, _ = held_out_ll_qmatrix(wide_aa, item_dim_joint, unique_joint,
                                          n_folds=args.n_folds, epochs=args.epochs, seed=42)
    print(f"Joint CV LL/cell = {cv_ll_joint:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    cv_existing = pd.read_csv(in_dir / "qmatrix_cv_ll.csv")
    print("\nExisting CV LLs:", flush=True)
    print(cv_existing.to_string(index=False), flush=True)

    bic_K1 = 298021  # 7-model panel K=1 BIC, see factor_selection_metrics.csv
    rows = []
    for _, row in cv_existing.iterrows():
        rows.append({
            "variant": row["variant"], "K": int(row["K"]),
            "BIC": None, "delta_bic": None,
            "cv_ll_per_cell": row["cv_ll_per_cell"], "delta_cv_ll": row["delta_vs_K1"],
        })
    rows.append({
        "variant": "Q-matrix Joint Action×Skill",
        "K": len(unique_joint),
        "BIC": bic_joint, "delta_bic": bic_joint - bic_K1,
        "cv_ll_per_cell": cv_ll_joint, "delta_cv_ll": cv_ll_joint - (-0.8834),
    })
    df = pd.DataFrame(rows)
    df.to_csv(in_dir / "qmatrix_within_action_comparison.csv", index=False)
    print(f"\nFinal comparison:", flush=True)
    print(df.to_string(index=False), flush=True)
    print(f"\nsaved {in_dir / 'qmatrix_within_action_comparison.csv'}", flush=True)

    # Verdict
    cv_action = cv_existing[cv_existing["variant"] == "Q-matrix Action GTO"]["cv_ll_per_cell"].iloc[0]
    cv_skill = cv_existing[cv_existing["variant"] == "Q-matrix Skill Heuristic"]["cv_ll_per_cell"].iloc[0]
    print(f"\nVerdict:", flush=True)
    print(f"  Joint CV LL = {cv_ll_joint:.4f}", flush=True)
    print(f"  Action-only = {cv_action:.4f}", flush=True)
    print(f"  Skill-only  = {cv_skill:.4f}", flush=True)
    if cv_ll_joint > max(cv_action, cv_skill) + 0.005:
        print(f"  → Joint MEANINGFULLY BEATS both. Skill carries signal BEYOND action class.", flush=True)
    elif cv_ll_joint > max(cv_action, cv_skill):
        print(f"  → Joint marginally beats both. Skill adds modest signal beyond action.", flush=True)
    else:
        print(f"  → Joint does NOT beat both. Skill labels do NOT add signal beyond action class.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
