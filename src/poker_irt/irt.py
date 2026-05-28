"""1PL (Rasch) and 2PL unidimensional IRT fitting via py-irt."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


@dataclass
class IrtFit:
    model_type: str                    # "1pl" or "2pl"
    ability: dict[str, float]
    difficulty: dict[str, float]
    discrimination: dict[str, float]   # all 1.0 for 1PL
    item_order: list[str]
    subject_order: list[str]
    n_items: int
    n_subjects: int

    def ability_series(self) -> pd.Series:
        return pd.Series(self.ability).sort_values(ascending=False)

    def difficulty_series(self) -> pd.Series:
        return pd.Series(self.difficulty)

    def discrimination_series(self) -> pd.Series:
        return pd.Series(self.discrimination)


def _wide_to_jsonlines(wide: pd.DataFrame, out_path: Path) -> None:
    """Write py-irt input: one line per subject with a ``responses`` dict."""
    with out_path.open("w") as f:
        for subject_id in wide.columns:
            col = wide[subject_id].dropna()
            responses = {item_id: int(v) for item_id, v in col.items()}
            f.write(json.dumps({
                "subject_id": str(subject_id),
                "responses": responses,
            }) + "\n")


def fit_irt(
    wide_aa: pd.DataFrame,
    model_type: str = "2pl",
    epochs: int = 2000,
    seed: int = 42,
    lr: float = 0.1,
    verbose: bool = False,
) -> IrtFit:
    """Fit a 1PL or 2PL IRT model on the wide AA matrix using py-irt."""
    from py_irt.config import IrtConfig
    from py_irt.training import IrtModelTrainer

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonlines", delete=False) as f:
        path = Path(f.name)
    try:
        _wide_to_jsonlines(wide_aa, path)
        cfg = IrtConfig(
            model_type=model_type, epochs=epochs, seed=seed, lr=lr, log_every=max(epochs // 5, 1)
        )
        trainer = IrtModelTrainer(data_path=path, config=cfg, verbose=verbose)
        trainer.train()
    finally:
        path.unlink(missing_ok=True)

    params = trainer.last_params
    subject_order = [params["subject_ids"][i] for i in sorted(params["subject_ids"])]
    item_order = [params["item_ids"][i] for i in sorted(params["item_ids"])]
    ability = dict(zip(subject_order, params["ability"]))
    difficulty = dict(zip(item_order, params["diff"]))
    if model_type == "2pl":
        discrimination = dict(zip(item_order, params["disc"]))
    else:
        discrimination = {i: 1.0 for i in item_order}

    return IrtFit(
        model_type=model_type,
        ability=ability,
        difficulty=difficulty,
        discrimination=discrimination,
        item_order=item_order,
        subject_order=subject_order,
        n_items=len(item_order),
        n_subjects=len(subject_order),
    )


def bootstrap_abilities(
    wide_aa: pd.DataFrame,
    model_type: str = "2pl",
    n_bootstrap: int = 100,
    epochs: int = 1000,
    base_seed: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """Resample items with replacement, refit, return one ability row per draw."""
    rows = []
    items = wide_aa.index.values
    rng = np.random.RandomState(base_seed)
    for b in range(n_bootstrap):
        idx = rng.choice(items, size=len(items), replace=True)
        resampled = wide_aa.loc[idx].reset_index(drop=True)
        resampled.index = [f"boot_{b}_item_{i}" for i in range(len(resampled))]
        fit = fit_irt(resampled, model_type=model_type, epochs=epochs,
                      seed=base_seed + b, verbose=verbose)
        rows.append(fit.ability)
    return pd.DataFrame(rows)


def test_information(fit: IrtFit, theta_grid: np.ndarray) -> np.ndarray:
    """Test information function I(theta) over a grid of theta values."""
    alpha = np.array([fit.discrimination[i] for i in fit.item_order])
    beta = np.array([fit.difficulty[i] for i in fit.item_order])
    info = np.zeros_like(theta_grid, dtype=float)
    for k, theta in enumerate(theta_grid):
        p = 1.0 / (1.0 + np.exp(-alpha * (theta - beta)))
        info[k] = float(np.sum(alpha ** 2 * p * (1 - p)))
    return info
