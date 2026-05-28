"""Paper figures for the 7-model run (outputs/main_11k_seed42_7models/figs/)."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

RUN = "main_11k_seed42_7models"
ART = Path(f"outputs/{RUN}/artifacts")
FIGS = Path(f"outputs/{RUN}/figs")
FIGS.mkdir(parents=True, exist_ok=True)


def short(m: str) -> str:
    if ":" in m: m = m.split(":", 1)[1]
    return m.split("/")[-1]


def fig3_qmatrix_bic():
    """Bar chart of ΔBIC vs the K=1 free factor baseline."""
    rows = [
        ("K=1 free factor", 0, "baseline"),
        ("Phase Q-matrix (2)", -4741, "no LLM"),
        ("Gemini skill (8)", -9099, "LLM labels"),
        ("Heuristic skill (8)", -10225, "no LLM"),
        ("Action GTO (4)", -10540, "no LLM"),
    ]
    df = pd.DataFrame(rows, columns=["model", "delta_bic", "labels"])
    df = df.sort_values("delta_bic", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    colors = {"baseline": "#999999", "no LLM": "#1f77b4", "LLM labels": "#ff7f0e"}
    bar_colors = [colors[l] for l in df["labels"]]
    bars = ax.barh(df["model"], df["delta_bic"], color=bar_colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, df["delta_bic"]):
        ax.text(v - 150 if v < 0 else 150, b.get_y() + b.get_height()/2,
                f"{v:+,d}", va="center", ha="right" if v < 0 else "left", fontsize=10)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("ΔBIC vs K=1 free factor   (more negative = better fit)")
    ax.set_title("All Q-matrices beat the unconstrained 1-factor model\n"
                 "(action and heuristic variants use NO LLM-derived labels)")
    handles = [plt.Rectangle((0,0),1,1, color=c, label=l) for l,c in colors.items()]
    ax.legend(handles=handles, loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(FIGS / "paper_fig3_qmatrix_bic.png", dpi=140)
    plt.close()
    print(f"saved {FIGS / 'paper_fig3_qmatrix_bic.png'}")


def fig4a_permutation():
    perm = pd.read_csv(ART / "permutation_test_skill.csv")
    real = -9099  # measured Gemini Q-matrix ΔBIC for the 7-model panel

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.hist(perm["delta_bic_null"], bins=10, color="#999999", edgecolor="black", label=f"Null (N={len(perm)})")
    ax.axvline(real, color="red", linestyle="--", linewidth=2, label=f"Real ΔBIC = {real:,}")
    ax.axvline(perm["delta_bic_null"].mean(), color="#1f77b4", linewidth=1,
               label=f"Null mean = {perm['delta_bic_null'].mean():,.0f}")
    ax.set_xlabel("ΔBIC (Q-matrix - K=1 free)")
    ax.set_ylabel("count")
    ax.set_title(f"Permutation null distribution for Gemini skill Q-matrix\n"
                 f"Real labels lie {abs(real - perm['delta_bic_null'].mean()):,.0f} BIC points beyond null mean — p < 1/{len(perm)}")
    ax.set_xlim(min(real - 500, perm["delta_bic_null"].min() - 100),
                perm["delta_bic_null"].max() + 100)
    ax.legend(loc="center left")
    plt.tight_layout()
    plt.savefig(FIGS / "paper_fig4a_permutation.png", dpi=140)
    plt.close()
    print(f"saved {FIGS / 'paper_fig4a_permutation.png'}")


def fig4b_kappa():
    cm = pd.read_csv(ART / "gemini_vs_heuristic_confusion.csv", index_col=0)
    cm_pct = cm.div(cm.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10.5, 6))
    sns.heatmap(cm_pct, annot=cm.values, fmt="d", cmap="Blues", ax=ax,
                cbar_kws={"label": "row-fraction"}, annot_kws={"size": 9})
    ax.set_xlabel("Heuristic label (no LLM)")
    ax.set_ylabel("Gemini label")
    ax.set_title(f"Gemini-vs-heuristic Q-matrix label agreement\nκ = 0.666 (substantial), 72.7% raw agreement (N=11,000 items)")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGS / "paper_fig4b_kappa.png", dpi=140)
    plt.close()
    print(f"saved {FIGS / 'paper_fig4b_kappa.png'}")


def fig5_ability_heatmap():
    """N x K ability heatmap (action profile preferred, skill fallback)."""
    candidates = ["qmatrix_profile_action.csv", "qmatrix_profile_skill_gemini.csv"]
    prof_path = None
    for c in candidates:
        if (ART / c).exists():
            prof_path = ART / c
            break
    if prof_path is None:
        print("  no profile csv found"); return
    prof = pd.read_csv(prof_path, index_col=0)
    prof.index = [short(m) for m in prof.index]

    fig, ax = plt.subplots(figsize=(max(8.5, 0.9 * prof.shape[1] + 4), 4.2))
    sns.heatmap(prof, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax,
                cbar_kws={"label": "ability"},
                annot_kws={"size": 10})
    ax.set_xlabel(prof_path.stem.replace("qmatrix_profile_", "") + " class")
    ax.set_ylabel("Model")
    ax.set_title(f"Q-matrix ability profile — {prof_path.stem}\n"
                 f"Same panel reorders along different skill axes")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGS / "paper_fig5_ability_heatmap.png", dpi=140)
    plt.close()
    print(f"saved {FIGS / 'paper_fig5_ability_heatmap.png'}")


def fig2_per_action_aa():
    aa_path = ART / "aa_by_action_class.csv"
    if not aa_path.exists():
        print(f"  skip {aa_path}"); return
    df = pd.read_csv(aa_path)
    print("per-action AA columns:", df.columns.tolist())
    if "model_id" in df.columns and "gto_action_class" in df.columns:
        val_col = "aa" if "aa" in df.columns else df.select_dtypes(include=[np.number]).columns[0]
        pivot = df.pivot(index="model_id", columns="gto_action_class", values=val_col)
    elif "model_id" in df.columns:
        pivot = df.set_index("model_id")
    else:
        pivot = df
    pivot.index = [short(m) for m in pivot.index]

    fig, ax = plt.subplots(figsize=(max(8.5, 1.0 * pivot.shape[1] + 4), 4.2))
    sns.heatmap(pivot, cmap="Blues", annot=True, fmt=".2f", ax=ax,
                cbar_kws={"label": "AA"}, annot_kws={"size": 10})
    ax.set_xlabel("GTO action class")
    ax.set_ylabel("Model")
    ax.set_title("Per-action accuracy on PokerBench (11K items)")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGS / "paper_fig2_per_action_aa.png", dpi=140)
    plt.close()
    print(f"saved {FIGS / 'paper_fig2_per_action_aa.png'}")


if __name__ == "__main__":
    fig3_qmatrix_bic()
    fig4a_permutation()
    fig4b_kappa()
    fig5_ability_heatmap()
    fig2_per_action_aa()
