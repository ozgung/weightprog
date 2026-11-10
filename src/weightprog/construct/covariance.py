"""Covariance-whitened key -> value updates of an MLP down_proj (the closed form behind ROME / MEMIT).

    C, n = second_moment(be, texts, layer)                 # E[k k^T] of the layer's MLP activations on ordinary text
    edit = kv_update(layer, K, R, C, lam=..., eps=...)     # W' = W + R^T A,  A = K (lam C + K^T K + eps I)^-1

- K [n, d_ff]: the MLP activations ("keys") at the rows where the edit must act (from `probe.collect_tail(...).act`).
- R [n, H]: what down_proj's output should gain at those rows (e.g. alpha * a direction from `logit_direction`, or a
  `residual_write_direction` for a middle layer).
- C [d_ff, d_ff]: the second moment of keys on text where the edit must stay quiet (thousands of positions).

The update solves min ||dW K^T - R^T||^2 + lam * E_k ||dW k||^2 over ordinary keys k, so it hits the wanted rows and is
small, by construction, in the directions ordinary text uses: the principled version of a gate's "~0 elsewhere".
Rank = n (at most 128 rows per update; stack several for more). `lam` trades fit on K against quiet elsewhere;
`eps` keeps the system well-posed when C has fewer positions than d_ff (E4B: 10,240 units).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..edits import ModuleEdit
from ..names import module_name


def second_moment(be, texts: Iterable[Sequence[int] | str], layer: int, bos: str | None = "<bos>",
                  stride: int = 1, skip: int = 1) -> tuple[np.ndarray, int]:
    """Sum of k k^T over positions of `texts` (token ids or strings) for layer `layer`'s MLP activations, divided by
    the number of positions. Returns (C [d_ff, d_ff] float64, n_positions). Memory: d_ff^2 * 8 bytes (840 MB on E4B)."""
    tok = be.tokenizer
    b = tok.token_to_id(bos) if bos else None
    C, n = None, 0
    for t in texts:
        ids = list(t) if not isinstance(t, str) else ([b] if b is not None else []) + tok.encode(t)
        pos = list(range(skip, len(ids), stride))
        if not pos:
            continue
        X = be.capture(ids, pos, layers=[layer], what={"mlp"})["mlp"][0].astype(np.float64)
        C = X.T @ X if C is None else C + X.T @ X
        n += len(pos)
    if C is None:
        raise ValueError("no positions")
    return C / n, n


def kv_update(layer: int, K: np.ndarray, R: np.ndarray, C: np.ndarray, lam: float = 1.0, eps: float = 1e-4,
              note: str = "", null_basis: np.ndarray | None = None) -> ModuleEdit:
    """down_proj edit with dW = R^T (K (lam C + K^T K + eps I)^-1): rank n = len(K). See the module docstring.
    With `null_basis` U [d_ff, k] (construct.null_space_basis of the inputs to preserve), the update is solved inside
    span(U) (AlphaEdit): inputs orthogonal to U are left exactly unchanged."""
    K = np.atleast_2d(np.asarray(K, dtype=np.float64))
    R = np.atleast_2d(np.asarray(R, dtype=np.float64))
    if K.shape[0] != R.shape[0]:
        raise ValueError("K and R need one row per target position")
    if K.shape[0] > 128:
        raise ValueError("at most 128 rows per update (rank limit); stack several updates instead")
    d = K.shape[1]
    if null_basis is None:
        M = lam * np.asarray(C, dtype=np.float64) + K.T @ K + eps * np.eye(d)
        A = np.linalg.solve(M, K.T).T                   # [n, d_ff]
    else:
        U = np.asarray(null_basis, dtype=np.float64)
        Ku = K @ U                                      # keys seen inside the null space [n, k]
        M = lam * (U.T @ np.asarray(C, dtype=np.float64) @ U) + Ku.T @ Ku + eps * np.eye(U.shape[1])
        A = np.linalg.solve(M, Ku.T).T @ U.T            # [n, d_ff], zero on the preserved span
    return ModuleEdit(module_name(layer, "down_proj"), R.T, A,
                      {"construct": "kv_update", "rank": int(K.shape[0]), "lam": lam, "eps": eps, "note": note,
                       "null_dim": None if null_basis is None else int(np.asarray(null_basis).shape[1])})
