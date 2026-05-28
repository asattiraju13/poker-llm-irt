"""Compare a Q-matrix-constrained model to the K=1 free factor baseline by BIC."""
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

from poker_irt.factor import fit_factor_model, factor_bic
from poker_irt.qmatrix import fit_qmatrix, qmatrix_bic
from poker_irt.logging_setup import setup_logging, write_metrics


def short(m: str) -> str:
    return m.split(":", 1)[-1].split("/")[-1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--dimension", choices=["action", "phase", "skill"], default="action")
    p.add_argument("--skill-file", default=None,
                   help="for --dimension skill: parquet with item_id + skill columns")
    args = p.parse_args()

    log = setup_logging(args.run, f"qmatrix_comparison_{args.dimension}")
    t0 = time.time()

    in_dir = Path(f"outputs/{args.run}/artifacts")
    long = pd.read_parquet(in_dir / "response_long.parquet")
    wide_aa = pd.read_parquet(in_dir / "response_wide_aa.parquet")

    with (in_dir / "factor_fits" / "K1.pkl").open("rb") as f:
        fit_uncon = pickle.load(f)
    bic_uncon = factor_bic(fit_uncon, wide_aa)
    log.info(f"Unconstrained K=1: LL={fit_uncon.log_lik:.2f}  BIC={bic_uncon:.2f}")

    if args.dimension == "action":
        item_dim_map_raw = long.drop_duplicates("item_id").set_index("item_id")["gto_action_class"]
        unique_dims = sorted(item_dim_map_raw.unique())
        dim_to_int = {d: i for i, d in enumerate(unique_dims)}
        item_dim_map = {iid: dim_to_int[d] for iid, d in item_dim_map_raw.items()}
        dim_names = unique_dims
    elif args.dimension == "skill":
        skill_path = args.skill_file or f"data/responses/{args.run}/item_skills_at_11k.parquet"
        skills = pd.read_parquet(skill_path).set_index("item_id")["skill"]
        wide_items = set(wide_aa.index)
        skills = skills[skills.index.isin(wide_items)]
        item_dim_map_raw = skills
        unique_dims = sorted(item_dim_map_raw.unique())
        dim_to_int = {d: i for i, d in enumerate(unique_dims)}
        item_dim_map = {iid: dim_to_int[d] for iid, d in item_dim_map_raw.items()}
        dim_names = unique_dims
    else:  # phase
        item_dim_map_raw = long.drop_duplicates("item_id").set_index("item_id")["split"]
        unique_dims = sorted(item_dim_map_raw.unique())
        dim_to_int = {d: i for i, d in enumerate(unique_dims)}
        item_dim_map = {iid: dim_to_int[d] for iid, d in item_dim_map_raw.items()}
        dim_names = unique_dims
    log.info(f"Q-matrix dimensions ({args.dimension}): {dim_names}")
    log.info(f"Items per dim: {pd.Series(list(item_dim_map.values())).map(dict(enumerate(dim_names))).value_counts().to_dict()}")

    log.info(f"Fitting Q-matrix K={len(dim_names)} (best of {args.n_seeds} seeds)")
    best = None
    for s in range(args.n_seeds):
        ts = time.time()
        fit_q = fit_qmatrix(wide_aa, item_dim_map, dim_names,
                            epochs=args.epochs, seed=42 + s)
        log.info(f"  seed {42+s}: LL={fit_q.log_lik:.2f}  ({time.time()-ts:.1f}s)")
        if best is None or fit_q.log_lik > best.log_lik:
            best = fit_q
    bic_q = qmatrix_bic(best, wide_aa)
    log.info(f"Best Q-matrix K={best.K}: LL={best.log_lik:.2f}  BIC={bic_q:.2f}")
    log.info("Per-subject Q-matrix abilities (rows=subject, cols=dim):")
    qprof = pd.DataFrame(best.U, index=best.subject_order, columns=dim_names)
    log.info("\n" + qprof.round(3).to_string())

    delta_bic = bic_q - bic_uncon
    delta_ll = best.log_lik - fit_uncon.log_lik
    winner = f"Q-matrix K={best.K} (constrained, {args.dimension})" if delta_bic < 0 else "Unconstrained K=1"
    log.info(f"ΔLL = {delta_ll:+.0f}  ΔBIC = {delta_bic:+.0f}  → {winner}")

    suffix = args.dimension
    with (in_dir / f"qmatrix_fit_{suffix}.pkl").open("wb") as f:
        pickle.dump(best, f)
    qprof.to_csv(in_dir / f"qmatrix_profile_{suffix}.csv")

    elapsed = time.time() - t0
    write_metrics(args.run, f"qmatrix_comparison_{args.dimension}", {
        "wall_time_s": round(elapsed, 2),
        "dimension": args.dimension,
        "uncon_ll": fit_uncon.log_lik, "uncon_bic": bic_uncon,
        "qmat_ll": best.log_lik, "qmat_bic": bic_q, "qmat_K": best.K,
        "delta_bic": delta_bic, "delta_ll": delta_ll, "winner": winner,
    })

    log.info(f"done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
