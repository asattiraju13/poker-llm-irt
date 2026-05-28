"""Q-matrix constrained multidimensional IRT.

Each item j has a known dimension k(j) in {0, ..., K-1}. The model is::

    P(Y_ij = 1) = sigmoid(U[i, k(j)] * V_j - z_j)

with U (n_subj x K) of subject abilities per dimension, V (n_item,) of scalar
item loadings on the assigned dimension, and z (n_item,) of item difficulty
offsets. Compared to a free K-factor model, the Q-matrix constraint adds only
(K-1) * n_subj parameters above K=1.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


@dataclass
class QMatrixFit:
    K: int
    U: np.ndarray             # (n_subj, K)
    V: np.ndarray             # (n_item,)
    z: np.ndarray             # (n_item,)
    item_dim: np.ndarray      # (n_item,) integer dimension index per item
    dim_names: list[str]
    subject_order: list[str]
    item_order: list[str]
    log_lik: float
    n_obs: int


def _qmatrix_model(Y, M, item_dim, K):
    n_subj, n_item = Y.shape
    U = pyro.sample("U", dist.Normal(torch.zeros(n_subj, K), torch.ones(n_subj, K)).to_event(2))
    V = pyro.sample("V", dist.Normal(torch.zeros(n_item), torch.ones(n_item)).to_event(1))
    z = pyro.sample("z", dist.Normal(torch.zeros(n_item), torch.ones(n_item) * 2.0).to_event(1))

    U_per_item = U[:, item_dim]
    logits = U_per_item * V.unsqueeze(0) - z.unsqueeze(0)

    with pyro.plate("subjects", n_subj):
        with pyro.plate("items", n_item):
            pyro.sample("y", dist.Bernoulli(logits=logits.t()).mask(M.t()), obs=Y.t())


def _qmatrix_guide(Y, M, item_dim, K):
    n_subj, n_item = Y.shape
    U_loc = pyro.param("U_loc", torch.randn(n_subj, K) * 0.1)
    V_loc = pyro.param("V_loc", torch.randn(n_item) * 0.1)
    z_loc = pyro.param("z_loc", torch.zeros(n_item))
    pyro.sample("U", dist.Delta(U_loc).to_event(2))
    pyro.sample("V", dist.Delta(V_loc).to_event(1))
    pyro.sample("z", dist.Delta(z_loc).to_event(1))


def fit_qmatrix(
    wide_aa: pd.DataFrame,
    item_dim_map: dict[str, int],
    dim_names: list[str],
    epochs: int = 3000,
    lr: float = 0.05,
    seed: int = 42,
) -> QMatrixFit:
    """Fit the Q-matrix-constrained IRT model via Pyro SVI."""
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    arr = wide_aa.values.T.astype(float)
    mask = ~np.isnan(arr)
    Y = torch.tensor(np.where(mask, arr, 0.0), dtype=torch.float32)
    M = torch.tensor(mask.astype(float), dtype=torch.float32)
    item_dim = np.array([item_dim_map[i] for i in wide_aa.index], dtype=np.int64)
    item_dim_t = torch.tensor(item_dim, dtype=torch.long)
    K = len(dim_names)

    svi = SVI(_qmatrix_model, _qmatrix_guide, Adam({"lr": lr}), loss=Trace_ELBO())
    for step in range(epochs):
        svi.step(Y, M, item_dim_t, K)

    U = pyro.param("U_loc").detach().numpy()
    V = pyro.param("V_loc").detach().numpy()
    z = pyro.param("z_loc").detach().numpy()

    # Float64 log-likelihood with explicit clipping to avoid log(0) NaN.
    Yn = Y.numpy().astype(np.float64)
    Mn = M.numpy().astype(np.float64)
    U = np.nan_to_num(U.astype(np.float64), nan=0.0, posinf=50.0, neginf=-50.0)
    V = np.nan_to_num(V.astype(np.float64), nan=0.0, posinf=50.0, neginf=-50.0)
    z = np.nan_to_num(z.astype(np.float64), nan=0.0, posinf=50.0, neginf=-50.0)
    U_per_item = U[:, item_dim]
    logits = U_per_item * V[None, :] - z[None, :]
    logits = np.clip(logits, -50.0, 50.0)
    p = 1.0 / (1.0 + np.exp(-logits))
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    ll = float(np.sum(Mn * (Yn * np.log(p) + (1.0 - Yn) * np.log(1.0 - p))))

    # Sign-orient each dimension's U column to correlate positively with AA on
    # items in that dimension; flipping U requires flipping V for those items.
    for k in range(K):
        item_idx_k = item_dim == k
        if not item_idx_k.any():
            continue
        aa_in_dim = np.nanmean(np.where(mask[:, item_idx_k], arr[:, item_idx_k], np.nan), axis=1)
        if np.corrcoef(U[:, k], aa_in_dim)[0, 1] < 0:
            U[:, k] = -U[:, k]
            V[item_idx_k] = -V[item_idx_k]

    return QMatrixFit(
        K=K, U=U, V=V, z=z, item_dim=item_dim, dim_names=dim_names,
        subject_order=list(wide_aa.columns), item_order=list(wide_aa.index),
        log_lik=ll, n_obs=int(M.sum().item()),
    )


def qmatrix_bic(fit: QMatrixFit, wide_aa: pd.DataFrame) -> float:
    """BIC for the Q-matrix model: p = K * n_subj + 2 * n_item."""
    n_subj = wide_aa.shape[1]
    n_item = wide_aa.shape[0]
    p = fit.K * n_subj + 2 * n_item
    return -2 * fit.log_lik + p * np.log(fit.n_obs)
