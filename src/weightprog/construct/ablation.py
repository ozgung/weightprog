"""Direction ablation as a low-rank edit.

Output side: remove the components of a module's output along orthonormal directions V [out, k]:
    W' = (I - V V^T) W        =>  B = -V,  A = V^T W           (rank k)
Input side: make the module blind to input directions U [in, k]:
    W' = W (I - U U^T)        =>  B = -W U,  A = U^T
"""

from __future__ import annotations

import numpy as np

from ..edits import ModuleEdit
from .linear import orthonormal


def ablate_output(module: str, W: np.ndarray, V: np.ndarray, strength: float = 1.0) -> ModuleEdit:
    """strength 1 removes the directions; 0 < strength < 1 shrinks them; strength 2 reflects them."""
    W = np.asarray(W, dtype=np.float64)
    V = orthonormal(np.asarray(V, dtype=np.float64).reshape(W.shape[0], -1))
    return ModuleEdit(module, -strength * V, V.T @ W, {"construct": "ablate_output", "rank": V.shape[1], "strength": strength})


def ablate_input(module: str, W: np.ndarray, U: np.ndarray, strength: float = 1.0) -> ModuleEdit:
    W = np.asarray(W, dtype=np.float64)
    U = orthonormal(np.asarray(U, dtype=np.float64).reshape(W.shape[1], -1))
    return ModuleEdit(module, -strength * (W @ U), U.T, {"construct": "ablate_input", "rank": U.shape[1], "strength": strength})


def quiet_units(W_down: np.ndarray, act_rms: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k MLP units contributing least (|down_proj column| x rms activation): candidates to rewire
    when an edit needs fresh units without disturbing the layer's existing function."""
    score = np.linalg.norm(np.asarray(W_down, dtype=np.float64), axis=0) * np.asarray(act_rms, dtype=np.float64)
    return np.argsort(score)[:k]
