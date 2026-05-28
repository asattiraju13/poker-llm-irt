"""K-factor sweep on the 7-model panel for the appendix table.

Sweeps K in {1..15} for in-sample BIC and 5-fold held-out LL for K in {1..4}.
Saves to ``outputs/main_11k_seed42_7models/artifacts/factor_selection_metrics_extended.csv``.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import pandas as pd

from poker_irt.factor import fit_factor_model, factor_bic, held_out_log_lik

RUN = "main_11k_seed42_7models"
ART = Path(f"outputs/{RUN}/artifacts")
ART.mkdir(parents=True, exist_ok=True)

wide_aa = pd.read_parquet(ART / "response_wide_aa.parquet")
n_subj = wide_aa.shape[1]; n_item = wide_aa.shape[0]
print(f"Loaded {n_item} items × {n_subj} models", flush=True)

K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
CV_K_VALUES = {1, 2, 3, 4}

rows = []
for K in K_VALUES:
    print(f"\n=== K={K} ===", flush=True)
    t0 = time.time()
    fit = fit_factor_model(wide_aa, K=K, epochs=3000, seed=42)
    bic = factor_bic(fit, wide_aa)
    fit_time = time.time() - t0
    print(f"  LL={fit.log_lik:.2f}  BIC={bic:.2f}  ({fit_time:.1f}s)", flush=True)

    cv_ll = None
    cv_time = None
    if K in CV_K_VALUES:
        t1 = time.time()
        cv_ll = held_out_log_lik(wide_aa, K=K, n_folds=5, epochs=3000, seed=42)
        cv_time = time.time() - t1
        print(f"  CV LL/cell = {cv_ll:.4f}  ({cv_time:.1f}s)", flush=True)

    params = K * (n_subj + n_item) + n_item
    rows.append({
        "K": K, "log_lik_in": fit.log_lik, "BIC": bic,
        "held_out_ll_per_cell": cv_ll, "n_params": params,
        "fit_time_s": round(fit_time, 2),
        "cv_time_s": round(cv_time, 2) if cv_time else None,
    })

df = pd.DataFrame(rows)
df.to_csv(ART / "factor_selection_metrics_extended.csv", index=False)
print(f"\nsaved {ART / 'factor_selection_metrics_extended.csv'}\n", flush=True)
print(df.to_string(index=False), flush=True)
