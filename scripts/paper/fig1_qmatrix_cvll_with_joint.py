"""Figure 1: Q-matrix held-out log-likelihood bar chart including the joint Action x Skill variant.

Reads CV LL values from ``outputs/<run>/artifacts/qmatrix_within_action_comparison.csv``
(produced by ``qmatrix_cv_ll.py`` + ``qmatrix_within_action.py``) and plots
per-cell ΔCV LL relative to the K=1 free-factor baseline.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUN = "main_11k_seed42_7models"
FIGS = Path(f"outputs/{RUN}/figs")
FIGS.mkdir(parents=True, exist_ok=True)

# Load CV LL table (one row per Q-matrix variant + the K=1 baseline).
cv_csv = Path(f"outputs/{RUN}/artifacts/qmatrix_within_action_comparison.csv")
cv_df = pd.read_csv(cv_csv).set_index("variant")

# Variant -> label / data-source flag for the plot.
# data_source: "no LLM" variants use only PokerBench labels (action class /
# game phase / heuristic skills); "LLM labels" uses Gemini-labeled skills.
display_rows = [
    ("K=1 free factor",            "K=1 free factor",                                "baseline"),
    ("Q-matrix Phase",             "Phase Q-matrix",                                 "no LLM"),
    ("Q-matrix Skill Gemini",      "Gemini skill (8)",                               "LLM labels"),
    ("Q-matrix Skill Heuristic",   "Heuristic skill (8)",                            "no LLM"),
    ("Q-matrix Action GTO",        "Action GTO (4)",                                 "no LLM"),
    ("Q-matrix Joint Action×Skill", r"$\mathbf{Joint\ Action \times Skill\ (11)}$",  "no LLM"),
]
rows = []
for key, label, source in display_rows:
    if key not in cv_df.index:
        raise SystemExit(f"missing variant in {cv_csv}: {key}")
    rows.append((label, float(cv_df.loc[key, "delta_cv_ll"]), source))

df = pd.DataFrame(rows, columns=["model", "delta_cv_ll", "labels"])
# Ascending order: smaller (worse) improvement on top, largest improvement at the bottom.
df = df.sort_values("delta_cv_ll", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(9.0, 4.8))
colors = {"baseline": "#999999", "no LLM": "#1f77b4", "LLM labels": "#ff7f0e"}
bar_colors = [colors[l] for l in df["labels"]]
bars = ax.barh(df["model"], df["delta_cv_ll"], color=bar_colors, edgecolor="black", linewidth=0.5)
for b, v in zip(bars, df["delta_cv_ll"]):
    # Place numeric label just outside the bar end.
    x_off = 0.003
    ax.text(v + x_off if v >= 0 else v - x_off,
            b.get_y() + b.get_height() / 2,
            f"{v:+.3f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=10)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel(r"$\Delta$ held-out log-likelihood per cell vs $K{=}1$ free factor "
              "(higher = better fit)")
ax.set_title("All confirmatory Q-matrices beat the unconstrained 1-factor model;\n"
             r"Joint Action$\times$Skill is the strongest variant (action and heuristic variants use NO LLM)")
# Pad x-axis a bit so labels don't clip.
xmax = df["delta_cv_ll"].max()
xmin = df["delta_cv_ll"].min()
pad = max(0.05, (xmax - xmin) * 0.15)
ax.set_xlim(min(xmin - pad, -pad), xmax + pad)
handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for l, c in colors.items()]
ax.legend(handles=handles, loc="lower right", frameon=True)
plt.tight_layout()
out = FIGS / "paper_fig1_qmatrix_cvll_with_joint.png"
plt.savefig(out, dpi=140)
plt.close()
print(f"saved {out}")
