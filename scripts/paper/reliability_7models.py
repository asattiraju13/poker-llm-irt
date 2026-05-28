"""Cronbach's alpha and split-half reliability for a wide AA matrix.

Reports alpha in both orientations (items-as-raters vs subjects-as-raters)
plus a Spearman-Brown-corrected split-half correlation, written to
``reliability_summary.json`` under the run's artifacts directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN = sys.argv[1] if len(sys.argv) > 1 else "main_11k_seed42_7models"
ART = Path(f"outputs/{RUN}/artifacts")

wide_aa = pd.read_parquet(ART / "response_wide_aa.parquet")
print(f"wide AA shape: {wide_aa.shape} (rows=items, cols=models)")

# Treat each item as a rater of each subject. The alpha formula is
# a = K/(K-1) * (1 - sum(var_col) / var_sum_row).
M = wide_aa.values  # rows=items, cols=subjects
N_items, K_subjects = M.shape
print(f"N items (raters) = {N_items}   K subjects (test items) = {K_subjects}")

col_var = M.var(axis=0, ddof=1)  # variance per subject across items
total_var = M.sum(axis=1).var(ddof=1)
alpha = K_subjects / (K_subjects - 1) * (1 - col_var.sum() / total_var)
print(f"Cronbach's alpha (subjects-as-items, items-as-raters): {alpha:.4f}")

# Alternative: items-as-test-items with subjects as raters.
M_T = M.T  # rows=subjects, cols=items
n_raters, k_test_items = M_T.shape
col_var_T = M_T.var(axis=0, ddof=1)
total_var_T = M_T.sum(axis=1).var(ddof=1)
alpha_T = k_test_items / (k_test_items - 1) * (1 - col_var_T.sum() / total_var_T)
print(f"Cronbach's alpha (items-as-test, subjects-as-raters): {alpha_T:.4f}")

# Split-half reliability via Spearman-Brown correction.
rng = np.random.RandomState(42)
idx = np.arange(N_items)
rng.shuffle(idx)
half = N_items // 2
h1 = idx[:half]
h2 = idx[half:half * 2]
sum_h1 = M[h1].sum(axis=0)
sum_h2 = M[h2].sum(axis=0)
r = np.corrcoef(sum_h1, sum_h2)[0, 1]
sb = 2 * r / (1 + r)
print(f"Split-half Pearson r: {r:.4f}")
print(f"Spearman-Brown corrected: {sb:.4f}")

print("\nPer-model AA:")
for m in wide_aa.columns:
    print(f"  {m}: {wide_aa[m].mean():.4f}")

out = {
    "n_items": int(N_items),
    "n_subjects": int(K_subjects),
    "cronbach_alpha_subjects_as_items": float(alpha),
    "cronbach_alpha_items_as_test": float(alpha_T),
    "split_half_r": float(r),
    "split_half_spearman_brown": float(sb),
    "per_model_aa": {m: float(wide_aa[m].mean()) for m in wide_aa.columns},
}
import json
(ART / "reliability_summary.json").write_text(json.dumps(out, indent=2))
print(f"\nsaved {ART / 'reliability_summary.json'}")
