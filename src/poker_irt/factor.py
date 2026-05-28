"""Multidimensional K-factor IRT model.

Model:
    P(Y_ij = 1 | u_i, v_j, z_j) = sigmoid(u_i^T v_j - z_j)

where u_i in R^K is subject i's K-dimensional ability, v_j in R^K is item j's
loading vector, and z_j is its difficulty offset. Fitted via Pyro SVI with
Normal priors and a delta variational guide (MAP point estimates). After
fitting, loadings are varimax-rotated and sign-oriented so each factor
correlates positively with mean per-subject accuracy.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


@dataclass
class FactorFit:
    K: int
    U: np.ndarray            # (n_subjects, K)
    V: np.ndarray            # (n_items, K)
    z: np.ndarray            # (n_items,)
    subject_order: list[str]
    item_order: list[str]
    log_lik: float
    n_obs: int
    elbo_history: list[float] = None


def _wide_to_tensor(wide_aa: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(Y, mask)`` tensors of shape ``(n_subjects, n_items)``."""
    arr = wide_aa.values.T.astype(float)
    mask = ~np.isnan(arr)
    Y = torch.tensor(np.where(mask, arr, 0.0), dtype=torch.float32)
    M = torch.tensor(mask.astype(float), dtype=torch.float32)
    return Y, M


def _model(Y: torch.Tensor, M: torch.Tensor, K: int):
    n_subj, n_item = Y.shape
    U = pyro.sample("U", dist.Normal(torch.zeros(n_subj, K), torch.ones(n_subj, K)).to_event(2))
    V = pyro.sample("V", dist.Normal(torch.zeros(n_item, K), torch.ones(n_item, K)).to_event(2))
    z = pyro.sample("z", dist.Normal(torch.zeros(n_item), torch.ones(n_item) * 2.0).to_event(1))
    logits = U @ V.t() - z.unsqueeze(0)
    with pyro.plate("subjects", n_subj):
        with pyro.plate("items", n_item):
            pyro.sample("y", dist.Bernoulli(logits=logits.t()).mask(M.t()), obs=Y.t())


def _guide(Y: torch.Tensor, M: torch.Tensor, K: int):
    n_subj, n_item = Y.shape
    U_loc = pyro.param("U_loc", torch.randn(n_subj, K) * 0.1)
    V_loc = pyro.param("V_loc", torch.randn(n_item, K) * 0.1)
    z_loc = pyro.param("z_loc", torch.zeros(n_item))
    pyro.sample("U", dist.Delta(U_loc).to_event(2))
    pyro.sample("V", dist.Delta(V_loc).to_event(2))
    pyro.sample("z", dist.Delta(z_loc).to_event(1))


def _post_fit_normalize(U: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean-center U columns; varimax-rotate (U, V) when K >= 2."""
    U = U - U.mean(axis=0, keepdims=True)
    if U.shape[1] >= 2:
        R = _varimax(V)
        V = V @ R
        U = U @ R
    return U, V


def _orient_to_aa(U: np.ndarray, V: np.ndarray, per_subj_aa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sign-flip each factor so its U column correlates positively with mean AA."""
    if len(per_subj_aa) <= 1:
        return U, V
    K = U.shape[1]
    for k in range(K):
        col = U[:, k]
        if np.std(col) < 1e-10:
            continue
        r = np.corrcoef(col, per_subj_aa)[0, 1]
        if r < 0:
            U[:, k] = -col
            V[:, k] = -V[:, k]
    return U, V


def _varimax(Phi: np.ndarray, gamma: float = 1.0, q: int = 50, tol: float = 1e-6) -> np.ndarray:
    """Compute the varimax rotation matrix for loadings ``Phi`` of shape (n, K)."""
    n, k = Phi.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(q):
        Lambda = Phi @ R
        u, s, vh = np.linalg.svd(
            Phi.T @ (Lambda ** 3 - (gamma / n) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda)))
        )
        R = u @ vh
        d_new = s.sum()
        if abs(d_new - d) < tol:
            break
        d = d_new
    return R


def fit_factor_model(
    wide_aa: pd.DataFrame,
    K: int,
    epochs: int = 2000,
    lr: float = 0.05,
    seed: int = 42,
    verbose: bool = False,
) -> FactorFit:
    """Fit the K-factor IRT model via Pyro SVI and return point estimates."""
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    Y, M = _wide_to_tensor(wide_aa)
    n_subj, n_item = Y.shape
    n_obs = int(M.sum().item())

    svi = SVI(_model, _guide, Adam({"lr": lr}), loss=Trace_ELBO())
    elbo_hist = []
    for step in range(epochs):
        loss = svi.step(Y, M, K)
        elbo_hist.append(loss)
        if verbose and (step + 1) % max(epochs // 5, 1) == 0:
            print(f"  step {step+1}/{epochs}  ELBO loss={loss:.2f}")

    U = pyro.param("U_loc").detach().numpy()
    V = pyro.param("V_loc").detach().numpy()
    z = pyro.param("z_loc").detach().numpy()
    U, V = _post_fit_normalize(U, V)
    per_subj_aa = wide_aa.mean(axis=0).values
    U, V = _orient_to_aa(U, V, per_subj_aa)

    # Compute final in-sample log-likelihood in float64 to avoid float32-epsilon
    # rounding 1 - 1e-8 to 1.0 and producing log(0) NaN.
    Yn = Y.numpy().astype(np.float64)
    Mn = M.numpy().astype(np.float64)
    U64 = np.nan_to_num(U.astype(np.float64), nan=0.0, posinf=50.0, neginf=-50.0)
    V64 = np.nan_to_num(V.astype(np.float64), nan=0.0, posinf=50.0, neginf=-50.0)
    z64 = np.nan_to_num(z.astype(np.float64), nan=0.0, posinf=50.0, neginf=-50.0)
    logits = U64 @ V64.T - z64[None, :]
    logits = np.clip(logits, -50.0, 50.0)
    p = 1.0 / (1.0 + np.exp(-logits))
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    ll = float(np.sum(Mn * (Yn * np.log(p) + (1.0 - Yn) * np.log(1.0 - p))))

    return FactorFit(
        K=K, U=U, V=V, z=z,
        subject_order=list(wide_aa.columns),
        item_order=list(wide_aa.index),
        log_lik=ll, n_obs=n_obs,
        elbo_history=elbo_hist,
    )


def factor_bic(fit: FactorFit, wide_aa: pd.DataFrame) -> float:
    """BIC for the K-factor model: p = K*n_subj + K*n_item + n_item."""
    n_subj = wide_aa.shape[1]
    n_item = wide_aa.shape[0]
    K = fit.K
    p = K * n_subj + K * n_item + n_item
    return -2 * fit.log_lik + p * np.log(fit.n_obs)


def held_out_log_lik(
    wide_aa: pd.DataFrame,
    K: int,
    n_folds: int = 5,
    epochs: int = 2000,
    lr: float = 0.05,
    seed: int = 42,
) -> float:
    """K-fold CV over response cells. Returns mean test log-likelihood per cell."""
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
        M_train = M_full.numpy().copy()
        for i, j in test_indices:
            M_train[i, j] = 0.0
        wide_train = wide_aa.copy()
        for i, j in test_indices:
            wide_train.iloc[j, i] = np.nan
        fit = fit_factor_model(wide_train, K=K, epochs=epochs, lr=lr, seed=seed + f)
        U64 = np.nan_to_num(fit.U.astype(np.float64), nan=0.0, posinf=50.0, neginf=-50.0)
        V64 = np.nan_to_num(fit.V.astype(np.float64), nan=0.0, posinf=50.0, neginf=-50.0)
        z64 = np.nan_to_num(fit.z.astype(np.float64), nan=0.0, posinf=50.0, neginf=-50.0)
        logits = U64 @ V64.T - z64[None, :]
        logits = np.clip(logits, -50.0, 50.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        p = np.clip(p, 1e-8, 1.0 - 1e-8)
        Yn = Y_full.numpy().astype(np.float64)
        for i, j in test_indices:
            y = Yn[i, j]
            total_ll += y * np.log(p[i, j]) + (1 - y) * np.log(1 - p[i, j])
        total_n += len(test_indices)

    return float(total_ll / max(total_n, 1))
