"""Real-model check 3: the closed-form tail model predicts the logit effect of a last-layer down_proj edit on E4B.

NOT run in CI. Needs the real model (either backend) and an edit on layers.41.mlp.down_proj (for example a
proposer's frozen edit; edits are model derivatives and are not shipped with this library).

    python scripts/validate_tail_prediction.py --backend mlx --model-dir M --calls C --edit RUN_OR_EDIT_DIR \\
        --n 8 --out /tmp/tail.json

For n calls it teacher-forces prompt + prefix + recorded old_string, captures layer 41's tail inputs (resid_mid,
mlp, pli) at the old_string positions, predicts the change of the top-20 base logits under the edit
(`predict_logit_shift`, exact and linear) and measures it with the edit applied. In float32 on the tiny test model
the exact prediction matches to 2e-4; on E4B in bf16 expect agreement to bf16 rounding (a few 0.01 logits) and a
linear prediction within roughly 10-20 % on the targeted tokens.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mlx", choices=["mlx", "torch"])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--calls", required=True)
    ap.add_argument("--edit", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from weightprog.backends import get_backend
    from weightprog.construct import TailState, predict_logit_shift
    from weightprog.edits import EditSet
    from weightprog.replay import load_calls

    be = get_backend(a.backend, a.model_dir, dtype="bfloat16", device=a.device)
    es = EditSet.load(a.edit)
    L = be.spec.n_layers - 1
    edit = es[f"model.language_model.layers.{L}.mlp.down_proj.weight"]
    rows = []
    for c in load_calls(a.calls)[: a.n]:
        tok = be.tokenizer
        p_ids, c_ids = tok.encode(c["prompt"] + c["prefix"]), tok.encode(c["old_string"])[:64]
        ids = p_ids + c_ids
        pos = list(range(len(p_ids) - 1, len(ids) - 1))
        base = be.capture(ids, pos, what={"logits"})["logits"]
        toks = sorted(set(np.argsort(-base, axis=1)[:, :20].ravel().tolist()))
        st = TailState.from_backend(be, ids, pos, L)
        with be.edits(EditSet([edit])):
            new = be.capture(ids, pos, what={"logits"})["logits"]
        meas = (new - base)[:, toks]
        ex = predict_logit_shift(st, edit, toks, "exact")
        li = predict_logit_shift(st, edit, toks, "linear")
        rows.append({"key": c["key"], "max_abs_measured": float(np.abs(meas).max()),
                     "exact_max_err": float(np.abs(ex - meas).max()), "linear_max_err": float(np.abs(li - meas).max()),
                     "linear_corr": float(np.corrcoef(li.ravel(), meas.ravel())[0, 1]) if meas.std() > 0 else None})
        print(rows[-1], flush=True)
    open(a.out, "w").write(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
