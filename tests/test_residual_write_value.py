"""post_norm_reach / residual_write_value: kv_update values that write a residual change through Gemma 4's post-norm."""

import numpy as np

from weightprog.construct import (kv_update, post_norm_reach, residual_write_direction, residual_write_value,
                                  second_moment)
from weightprog.construct.linear import rms_norm
from weightprog.edits import EditSet


def test_post_norm_reach_meets_constraint_and_beats_division():
    rng = np.random.default_rng(0)
    H = 64
    g = np.abs(rng.standard_normal(H)); g[:8] = 1e-3                 # a few tiny post-norm weights, as on E4B
    n = rng.standard_normal(H); n /= np.sqrt(np.mean(n ** 2))
    delta = 2.0 * rng.standard_normal(H)
    u = post_norm_reach(n, g, delta)
    assert abs(np.mean(u ** 2) - 1.0) < 1e-6                          # rms 1: the norm keeps it as is
    err = lambda v: np.linalg.norm(g * (v - n) - delta)               # noqa: E731
    naive = n + delta / g; naive /= np.sqrt(np.mean(naive ** 2))      # divide by the weight, then renormalise
    assert err(u) < err(naive) and err(u) < err(n)
    for _ in range(20):                                              # no random point on the sphere does better
        v = u + 0.05 * rng.standard_normal(H); v /= np.sqrt(np.mean(v ** 2))
        assert err(u) <= err(v) + 1e-9


def test_residual_write_value_through_kv_update(be):
    tok = be.tokenizer
    texts = ["class A:\n    def f(self, x):\n        return x", "import os\nprint(os.getcwd())",
             "def add(a, b):\n    return a + b", "for i in range(10):\n    total += i",
             "x = {'a': 1, 'b': 2}\ny = x.get", "with open(path) as f:\n    data = f.read"]
    seqs = [tok.encode(t) for t in texts]
    L = be.spec.n_layers // 2

    def last(what, es=None):
        out = {k: [] for k in what}
        for ids in seqs:                                             # one prompt per row: rows cannot see each other
            if es is None:
                cap = be.capture(ids, [len(ids) - 1], layers=[L], what=set(what))
            else:
                with be.edits(es):
                    cap = be.capture(ids, [len(ids) - 1], layers=[L], what=set(what))
            for k in what:
                out[k].append(cap[k][0, 0].astype(np.float64))
        return {k: np.stack(v) for k, v in out.items()}

    base = last(("resid", "mlp", "mlp_out"))
    rng = np.random.default_rng(1)
    r = rng.standard_normal(base["resid"].shape[1]); r /= np.linalg.norm(r)
    delta = 0.3 * np.linalg.norm(base["resid"], axis=1).mean() * r
    R = np.vstack([residual_write_value(be, L, base["mlp_out"][:3], delta), np.zeros((3, base["resid"].shape[1]))])
    C, _ = second_moment(be, seqs, L)
    es = EditSet([kv_update(L, base["mlp"], R, C, lam=1e-3, eps=1e-6)])
    ed = last(("resid", "mlp_out"), es)
    # exact part: the post-norm contribution s * w_post * rms_norm(mlp_out) moves as post_norm_reach solved
    w = be.read_weight(f"model.language_model.layers.{L}.post_feedforward_layernorm.weight").astype(np.float64)
    sc = float(np.asarray(be.read_weight(f"model.language_model.layers.{L}.layer_scalar")).reshape(-1)[0])
    contrib = lambda m: sc * rms_norm(m, w, be.spec.rms_eps)        # noqa: E731
    moved = contrib(ed["mlp_out"]) - contrib(base["mlp_out"])
    for i in range(3):
        rms = np.sqrt(np.mean(base["mlp_out"][i] ** 2) + be.spec.rms_eps)
        want = sc * w * (post_norm_reach(base["mlp_out"][i] / rms, sc * w, delta) - base["mlp_out"][i] / rms)
        np.testing.assert_allclose(moved[i], want, atol=1e-3 * np.linalg.norm(delta))
        assert want @ delta / (np.linalg.norm(want) * np.linalg.norm(delta)) > 0.85   # 64 dims, random direction
    assert np.abs(moved[3:]).max() < 0.05 * np.linalg.norm(delta)  # zero-value keys stay put
    # the residual after the layer: the per-layer-input branch reacts (large in the tiny random model), so loosely
    got = ed["resid"] - base["resid"]
    assert (got[:3] @ r).min() > 0.5 * np.linalg.norm(delta)
    # the old construction (direction / w_post, at the same value size) moves the residual less along r
    u = residual_write_direction(be, L, r, "mlp")
    R_old = R.copy(); R_old[:3] = np.linalg.norm(R[:3], axis=1)[:, None] * u
    got_old = last(("resid",), EditSet([kv_update(L, base["mlp"], R_old, C, lam=1e-3, eps=1e-6)]))["resid"] - base["resid"]
    assert (got[:3] @ r).mean() > (got_old[:3] @ r).mean()
