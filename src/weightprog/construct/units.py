"""Unit patching (rewire chosen MLP units into new detectors) and null-space projection of edits.

Unit patching, the family the study's editors used most (Opus: 40 and 80 rewired units in layer 41; Fable 5.2: the
same idea), close to T-Patcher's added neurons but in closed form and on existing low-use units:

    units = quiet_units(W_down, act_rms, k)                       # units that contribute least (construct.ablation)
    x = mlp_input(be, resid_mid, layer)                           # what gate_proj / up_proj read: pre-FFN normed
    es = unit_patch(be.read_weight, layer, units, gate_rows, up_rows, down_cols)

A rewired unit u computes gelu_tanh(gate_rows[u] . x) * (up_rows[u] . x) and writes it along down_cols[:, u]. A common
design: gate_rows = a ridge read-out of the condition (pre-activation, large where the behaviour should change),
up_rows = a read-out that is ~1 everywhere (fit to 1 on ordinary rows), down_cols = a logit direction
(`logit_direction`) scaled to the push you want. The old function of those units is lost, so pick quiet ones.

Null-space projection (AlphaEdit, Fang et al., ICLR 2025): restrict an edit to directions of the input space that
preserved inputs do not use, so it leaves them exactly unchanged:

    U = null_space_basis(C_keep, rel_tol=1e-3)     # C_keep: second moment of inputs to preserve (second_moment)
    edit2 = project_edit(edit, U)                   # A <- A U U^T : (W + B A U U^T) k = W k for k in span(C_keep)
    kv = kv_update(layer, K, R, C, null_basis=U)    # or solve the key -> value update inside the null space
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from ..edits import EditSet, ModuleEdit
from ..names import module_name
from .linear import rms_norm

PRE = "model.language_model."


def mlp_input(be, resid_mid: np.ndarray, layer: int) -> np.ndarray:
    """The MLP's input x = pre_feedforward_layernorm(resid_mid) for captured resid_mid rows [P, H]."""
    w = be.read_weight(f"{PRE}layers.{layer}.pre_feedforward_layernorm.weight").astype(np.float64)
    return rms_norm(np.asarray(resid_mid, dtype=np.float64), w, be.spec.rms_eps)


def unit_patch(read_weight: Callable[[str], np.ndarray], layer: int, units: Sequence[int], gate_rows: np.ndarray,
               up_rows: np.ndarray, down_cols: np.ndarray, note: str = "") -> EditSet:
    """Rewire MLP units of `layer`: their gate_proj and up_proj rows become gate_rows / up_rows [k, H] and their
    down_proj columns become down_cols [H, k]. Returns three rank-k module edits (exact replacements)."""
    u = np.asarray(units, dtype=np.int64)
    k = len(u)
    if not 1 <= k <= 128:
        raise ValueError("1 to 128 units per patch")
    gname, uname, dname = (module_name(layer, p) for p in ("gate_proj", "up_proj", "down_proj"))
    Wg, Wu, Wd = (np.asarray(read_weight(n), dtype=np.float64) for n in (gname, uname, dname))
    d_ff = Wg.shape[0]
    E = np.zeros((d_ff, k))
    E[u, np.arange(k)] = 1.0
    G = np.atleast_2d(np.asarray(gate_rows, dtype=np.float64))
    Up = np.atleast_2d(np.asarray(up_rows, dtype=np.float64))
    D = np.asarray(down_cols, dtype=np.float64).reshape(Wd.shape[0], k)
    meta = {"construct": "unit_patch", "units": u.tolist(), "note": note}
    return EditSet([ModuleEdit(gname, E, G - Wg[u], dict(meta)),
                    ModuleEdit(uname, E, Up - Wu[u], dict(meta)),
                    ModuleEdit(dname, D - Wd[:, u], E.T, dict(meta))])


def null_space_basis(C: np.ndarray, rel_tol: float = 1e-3, max_dim: int | None = None) -> np.ndarray:
    """Orthonormal basis U [d, k] of the directions where the second moment C [d, d] of preserved inputs has
    eigenvalue below rel_tol * the largest one (the inputs to preserve barely use them)."""
    w, V = np.linalg.eigh(np.asarray(C, dtype=np.float64))
    keep = w < rel_tol * w.max()
    U = V[:, keep]
    if max_dim is not None and U.shape[1] > max_dim:
        U = U[:, :max_dim]                          # eigh sorts ascending: the least-used directions first
    if U.shape[1] == 0:
        raise ValueError("no null space at this tolerance: raise rel_tol or use fewer preserved inputs")
    return U


def project_edit(edit: ModuleEdit, U: np.ndarray) -> ModuleEdit:
    """Restrict an edit to the input subspace spanned by U: A <- (A U) U^T, so inputs orthogonal to U are unchanged."""
    U = np.asarray(U, dtype=np.float64)
    if U.shape[0] != edit.A.shape[1]:
        raise ValueError(f"basis has {U.shape[0]} rows, the edit's input size is {edit.A.shape[1]}")
    return ModuleEdit(edit.module, edit.B.copy(), (edit.A.astype(np.float64) @ U) @ U.T,
                      dict(edit.meta, projected="null_space", null_dim=int(U.shape[1])))
