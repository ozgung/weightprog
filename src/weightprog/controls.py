"""Matched random controls for an edit: the size-matched null an evaluation compares against.

For every module of the source edit (B [out, r], A [r, in]) draw Gaussian B', A' of the same shape and scale both by
sqrt(||BA||_F / ||B'A'||_F), so the control has the same modules, the same rank and the same per-module Frobenius norm
of its delta. With support_matched=True the random B is zero on the rows where the source B is zero (a sensitivity
control for edits that touch few output coordinates).

Seeds: one numpy Generator per control, consumed module by module in file-name order, exactly as the original
`gate_controls.py`; the round-2 gate used seed = SEED_BASE + 10 * i + k for candidate i (1-based, in the order the
candidates were passed) and control k (1, 2).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .edits import EditSet, ModuleEdit


def matched_control(src: EditSet, seed: int, support_matched: bool = False) -> EditSet:
    rng = np.random.default_rng(seed)
    out = EditSet()
    for e in src:  # sorted by file name, as the original globbed sorted(edit/*.npz)
        B, A = e.B.astype(np.float32), e.A.astype(np.float32)
        target = float(np.linalg.norm(B @ A))
        Br = rng.standard_normal(B.shape).astype(np.float32)
        if support_matched:
            Br[np.abs(B).sum(1) == 0] = 0.0
        Ar = rng.standard_normal(A.shape).astype(np.float32)
        k = np.sqrt(target / float(np.linalg.norm(Br @ Ar)))
        Br *= k
        Ar *= k
        meta = {"module": e.module, "rank": int(B.shape[1]), "target_fro": target,
                "control_fro": float(np.linalg.norm(Br @ Ar)), "seed": seed, "support_matched": bool(support_matched),
                "nonzero_rows": int((np.abs(Br).sum(1) > 0).sum()), "source": e.meta.get("source", "")}
        out.add(ModuleEdit(e.module, Br, Ar, meta))
    return out


def write_control(src_dir: str | Path, out_dir: str | Path, seed: int, support_matched: bool = False) -> list[dict]:
    """CLI helper: read SRC/edit (or SRC), write OUT/edit/*.npz and OUT/control.json; return the meta records."""
    src = EditSet.load(src_dir)
    ctrl = matched_control(src, seed, support_matched)
    out = Path(out_dir)
    ctrl.save_dir(out / "edit")
    rec = [e.meta for e in ctrl]
    (out / "control.json").write_text(json.dumps(rec, indent=1))
    return rec


def control_seeds(seed_base: int, n_candidates: int, n_controls: int = 2) -> dict[tuple[int, int], int]:
    """{(i, k): seed_base + 10 i + k} for candidates i = 1..n and controls k = 1..n_controls (round-2 convention)."""
    return {(i, k): seed_base + 10 * i + k for i in range(1, n_candidates + 1) for k in range(1, n_controls + 1)}
