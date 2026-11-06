"""Closed-form solvers used by the constructors (no iterative fitting, no gradients)."""

from __future__ import annotations

import numpy as np


def ridge(X: np.ndarray, y: np.ndarray, lam: float = 1.0, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted ridge regression without intercept: argmin_a sum_i w_i (x_i . a - y_i)^2 + lam |a|^2.

    Solved in whichever form is smaller: primal (X^T W X + lam I)^-1 X^T W y when n >= d, else the dual
    a = X^T (W X X^T + lam I)^-1 W y. X [n, d], y [n] -> a [d]."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=np.float64)
    n, d = X.shape
    if n >= d:
        return np.linalg.solve(X.T @ (X * w[:, None]) + lam * np.eye(d), X.T @ (w * y))
    K = X @ X.T
    c = np.linalg.solve(w[:, None] * K + lam * np.eye(n), w * y)
    return X.T @ c


def min_norm(G: np.ndarray, t: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Minimum-norm x with G x ~= t: x = G^T (G G^T + eps I)^-1 t. G [k, d], t [k] -> x [d]."""
    G = np.asarray(G, dtype=np.float64)
    return G.T @ np.linalg.solve(G @ G.T + eps * np.eye(G.shape[0]), np.asarray(t, dtype=np.float64))


def orthonormal(V: np.ndarray) -> np.ndarray:
    """Columns of V orthonormalised (QR), V [d, k] -> [d, k]."""
    Q, _ = np.linalg.qr(np.asarray(V, dtype=np.float64))
    return Q


def rms_norm(x: np.ndarray, w: np.ndarray | None = None, eps: float = 1e-6) -> np.ndarray:
    """Gemma 4 RMSNorm: x / sqrt(mean(x^2) + eps) * w (the weight multiplies directly; no 1 + w)."""
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return y if w is None else y * w


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
