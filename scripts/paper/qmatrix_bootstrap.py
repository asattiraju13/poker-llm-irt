"""Bootstrap percentile CIs on Q-matrix subject ability profiles."""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from poker_irt.qmatrix import fit_qmatrix


def short(m: str) -> str:
    if ":" in m: m = m.split(":", 1)[1]
    return m.split("/")[-1]


def bootstrap_one_variant(wide_aa, item_dim_map, dim_names, n_boot, epochs, seed_base):
    """Bootstrap items with replacement, refit, return stacked U matrices."""
    n_item = wide_aa.shape[0]
    item_ids = list(wide_aa.index)
    Us = []
    for b in range(n_boot):
        rng = np.random.RandomState(seed_base + b)
        sample_idx = rng.choice(n_item, size=n_item, replace=True)
        new_rows = []
        new_ids = []
        new_dim_map = {}
        for new_pos, orig_pos in enumerate(sample_idx):
            orig_id = item_ids[orig_pos]
            new_id = f"{orig_id}__b{new_pos}"
            new_rows.append(wide_aa.iloc[orig_pos].values)
            new_ids.append(new_id)
            new_dim_map[new_id] = item_dim_map[orig_id]
        boot_wide = pd.DataFrame(new_rows, index=new_ids, columns=wide_aa.columns)
        t0 = time.time()
        fit = fit_qmatrix(boot_wide, new_dim_map, dim_names, epochs=epochs, seed=seed_base + b)
        Us.append(fit.U)
        print(f"  boot {b+1}/{n_boot}: LL={fit.log_lik:.0f}  ({time.time()-t0:.1f}s)", flush=True)
    return np.stack(Us, axis=0), fit.subject_order  # shape (n_boot, n_subj, K)


def run_variant(wide_aa, item_dim_map, dim_names, name, args, ART, FIGS):
    print(f"\n=== Bootstrapping {name} ({len(dim_names)} dims) ===", flush=True)
    t0 = time.time()
    Us, subject_order = bootstrap_one_variant(
        wide_aa, item_dim_map, dim_names,
        n_boot=args.n_boot, epochs=args.epochs, seed_base=42)
    print(f"  {args.n_boot} resamples done in {time.time()-t0:.1f}s", flush=True)

    mean = Us.mean(axis=0)
    lo = np.percentile(Us, 2.5, axis=0)
    hi = np.percentile(Us, 97.5, axis=0)
    std = Us.std(axis=0)

    rows = []
    for i, subj in enumerate(subject_order):
        for k, dn in enumerate(dim_names):
            rows.append({
                "subject": subj, "skill": dn,
                "mean": mean[i, k], "lo": lo[i, k], "hi": hi[i, k],
                "std": std[i, k], "width": hi[i, k] - lo[i, k],
            })
    df = pd.DataFrame(rows)
    out_csv = ART / f"qmatrix_bootstrap_{name}_profile_ci.csv"
    df.to_csv(out_csv, index=False)
    print(f"  saved {out_csv}", flush=True)

    mean_df = pd.DataFrame(mean, index=[short(s) for s in subject_order], columns=dim_names)
    lo_df = pd.DataFrame(lo, index=[short(s) for s in subject_order], columns=dim_names)
    hi_df = pd.DataFrame(hi, index=[short(s) for s in subject_order], columns=dim_names)

    fig, ax = plt.subplots(figsize=(max(8.5, 0.95 * len(dim_names) + 4), 5.0))
    sns.heatmap(mean_df, cmap="RdBu_r", center=0,
                annot=False, ax=ax,
                cbar_kws={"label": "ability (mean across bootstraps)"})
    for i in range(mean_df.shape[0]):
        for k in range(mean_df.shape[1]):
            m = mean_df.iloc[i, k]
            l = lo_df.iloc[i, k]
            h = hi_df.iloc[i, k]
            txt = f"{m:+.2f}\n[{l:+.1f},{h:+.1f}]"
            ax.text(k + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=8,
                    color="black" if abs(m) < 4 else "white")
    ax.set_title(f"{name} Q-matrix ability profile with 95% bootstrap CIs (N={args.n_boot})")
    ax.set_xlabel(name + " class"); ax.set_ylabel("model")
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    figpath = FIGS / f"qmatrix_profile_{name}_with_ci.png"
    plt.savefig(figpath, dpi=140)
    plt.close()
    print(f"  saved {figpath}", flush=True)

    # Pairs of subjects with non-overlapping bootstrap CIs per dimension.
    sig_rows = []
    for k, dn in enumerate(dim_names):
        for i in range(len(subject_order)):
            for j in range(i+1, len(subject_order)):
                if hi[i, k] < lo[j, k]:
                    sig_rows.append({"skill": dn, "subject_a": short(subject_order[i]),
                                    "subject_b": short(subject_order[j]),
                                    "winner": short(subject_order[j]),
                                    "a_mean": mean[i, k], "b_mean": mean[j, k]})
                elif hi[j, k] < lo[i, k]:
                    sig_rows.append({"skill": dn, "subject_a": short(subject_order[i]),
                                    "subject_b": short(subject_order[j]),
                                    "winner": short(subject_order[i]),
                                    "a_mean": mean[i, k], "b_mean": mean[j, k]})
    sig_df = pd.DataFrame(sig_rows)
    sig_path = ART / f"qmatrix_bootstrap_{name}_sig_pairs.csv"
    sig_df.to_csv(sig_path, index=False)
    print(f"  saved {sig_path} ({len(sig_df)} significant pairs)", flush=True)

    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="main_11k_seed42_7models")
    p.add_argument("--n-boot", type=int, default=80)
    p.add_argument("--epochs", type=int, default=2000,
                   help="fewer epochs are acceptable since we average many resamples")
    p.add_argument("--variants", nargs="+",
                   default=["action", "heuristic", "gemini"])
    p.add_argument("--gemini-skill-file",
                   default="data/responses/main_11k_seed42_7models/item_skills_at_11k.parquet")
    p.add_argument("--heuristic-skill-file",
                   default="data/responses/main_11k_seed42_7models/item_skills_HEURISTIC.parquet")
    args = p.parse_args()

    ART = Path(f"outputs/{args.run}/artifacts")
    FIGS = Path(f"outputs/{args.run}/figs")
    FIGS.mkdir(parents=True, exist_ok=True)

    wide_aa = pd.read_parquet(ART / "response_wide_aa.parquet")
    long = pd.read_parquet(ART / "response_long.parquet")
    wide_items = set(wide_aa.index)
    print(f"Loaded {wide_aa.shape[0]} items × {wide_aa.shape[1]} models", flush=True)

    if "action" in args.variants:
        item_dim_raw = long.drop_duplicates("item_id").set_index("item_id")["gto_action_class"]
        dims = sorted(item_dim_raw.unique())
        d2i = {d: i for i, d in enumerate(dims)}
        item_dim = {iid: d2i[d] for iid, d in item_dim_raw.items()}
        run_variant(wide_aa, item_dim, dims, "action", args, ART, FIGS)

    if "heuristic" in args.variants:
        s = pd.read_parquet(args.heuristic_skill_file).set_index("item_id")["skill"]
        s = s[s.index.isin(wide_items)]
        dims = sorted(s.unique())
        d2i = {d: i for i, d in enumerate(dims)}
        item_dim = {iid: d2i[d] for iid, d in s.items()}
        run_variant(wide_aa, item_dim, dims, "heuristic", args, ART, FIGS)

    if "gemini" in args.variants:
        s = pd.read_parquet(args.gemini_skill_file).set_index("item_id")["skill"]
        s = s[s.index.isin(wide_items)]
        dims = sorted(s.unique())
        d2i = {d: i for i, d in enumerate(dims)}
        item_dim = {iid: d2i[d] for iid, d in s.items()}
        run_variant(wide_aa, item_dim, dims, "gemini", args, ART, FIGS)

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
