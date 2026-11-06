"""Constructors on the tiny model: predicted logit shifts match the measured ones."""

import numpy as np
import pytest

from weightprog.construct import (LogitPath, TailState, ablate_input, ablate_output, fit_gate, gate_direction_edit,
                                  gate_values, logit_direction, predict_logit_shift, quiet_units, ridge, unit_gain)
from weightprog.edits import EditSet
from weightprog.names import module_name


def test_ridge_primal_equals_dual():
    rng = np.random.default_rng(0)
    X, y, w = rng.standard_normal((30, 12)), rng.standard_normal(30), rng.uniform(0.1, 2, 30)
    a1 = ridge(X, y, 0.7, w)
    a2 = ridge(X[:8], y[:8], 0.7, w[:8])  # n < d: dual
    a2p = np.linalg.solve(X[:8].T @ (X[:8] * w[:8, None]) + 0.7 * np.eye(12), X[:8].T @ (w[:8] * y[:8]))
    np.testing.assert_allclose(a2, a2p, atol=1e-10)
    np.testing.assert_allclose(a1, np.linalg.solve(X.T @ (X * w[:, None]) + 0.7 * np.eye(12), X.T @ (w * y)), atol=1e-10)


def test_ablation_edits():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((10, 7))
    v = rng.standard_normal(10)
    e = ablate_output("model.language_model.layers.0.mlp.down_proj.weight", W, v)
    W2 = W + e.delta()
    np.testing.assert_allclose(v @ W2, 0, atol=1e-5)  # no output along v
    u = rng.standard_normal(7)
    e2 = ablate_input("model.language_model.layers.0.mlp.down_proj.weight", W, u)
    np.testing.assert_allclose((W + e2.delta()) @ u, 0, atol=1e-5)  # blind to u
    assert list(quiet_units(np.diag([3.0, 1.0, 2.0]), np.ones(3), 2)) == [1, 2]


@pytest.fixture
def steer_setup(be, sample_ids):
    L = be.spec.n_layers - 1
    pos = list(range(5, len(sample_ids)))
    st = TailState.from_backend(be, sample_ids, pos, L)
    path = st.path
    # tokens to push down / up / hold
    down, up = [be.tokenizer.token_to_id(c) for c in ("e", "t")], [be.tokenizer.token_to_id(c) for c in ("x", "z")]
    hold = [be.tokenizer.token_to_id(c) for c in (" ", "\n", "a", "o", "(", ")")]
    targets = {**{t: -1.0 for t in down}, **{t: 0.5 for t in up}, **{t: 0.0 for t in hold}}
    d = logit_direction(path, targets)
    # gate: on at even positions, off at odd ones (closed-form ridge on the captured activations)
    y = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(len(pos))])
    a = fit_gate(st.act, y, lam=1e-3)
    return st, d, a, pos, down + up + hold, targets


def test_logit_direction_solves_targets(be, steer_setup):
    st, d, a, pos, toks, targets = steer_setup
    G = st.path.E_rows(toks) * (st.path.w_final * st.path.w_post)
    np.testing.assert_allclose(G @ d, [targets[t] for t in toks], atol=1e-4)
    np.testing.assert_allclose(st.act @ a, [1.0 if i % 2 == 0 else 0.0 for i in range(len(pos))], atol=0.05)


def test_predicted_shift_matches_measured(be, sample_ids, steer_setup):
    st, d, a, pos, toks, targets = steer_setup
    L = st.layer
    gain = unit_gain(st).mean()
    edit = gate_direction_edit(L, a[None], d[None], alpha=0.5 / gain)  # aim at ~0.5 logits per unit target
    base = be.capture(sample_ids, pos, what={"logits"})["logits"][:, toks]
    with be.edits(EditSet([edit])):
        new = be.capture(sample_ids, pos, what={"logits"})["logits"][:, toks]
    measured = new - base
    exact = predict_logit_shift(st, edit, toks, "exact")
    np.testing.assert_allclose(exact, measured, atol=2e-4)
    lin = predict_logit_shift(st, edit, toks, "linear")[:, :4]  # first order, on the targeted tokens
    mt = measured[:, :4]
    big = np.abs(mt) > 0.05
    assert big.sum() > 10
    rel = np.abs(lin[big] - mt[big]) / np.abs(mt[big])
    assert np.median(rel) < 0.2 and np.corrcoef(lin.ravel(), mt.ravel())[0, 1] > 0.95
    # selectivity: where the gate is ~0 the logits barely move
    g = gate_values(st, edit)[:, 0]
    off = np.abs(g) < 0.05
    assert np.abs(measured[off]).max() < 0.1 * np.abs(measured[~off]).max()
    # direction: the "down" tokens fall and the "up" tokens rise where the gate is on
    on = g > 0.9
    assert (measured[on][:, :2] < 0).mean() > 0.9 and (measured[on][:, 2:4] > 0).mean() > 0.9


def test_layer_scalar_pitfall(be, steer_setup):
    """Adding the pre-scalar MLP change to the post-scalar residual capture without the factor s overstates the
    effect; with s it matches the exact tail (the study's 0.48 calibration on E4B layer 41)."""
    from weightprog.construct import edit_delta, rms_norm

    st, d, a, pos, toks, targets = steer_setup
    s = st.path.scalar
    assert s == pytest.approx(0.48)
    edit = gate_direction_edit(st.layer, a[None], d[None], alpha=0.3 / unit_gain(st).mean())
    h = st.final_hidden()  # what a resid capture gives (post-scalar)
    m, dl = st.mlp_out(), edit_delta(st, edit)
    dn = rms_norm(m + dl, st.path.w_post) - rms_norm(m, st.path.w_post)
    true_dh = st.final_hidden(dl) - h
    naive_dh, scaled_dh = dn, s * dn
    # the per-layer-input branch feeds the change back (strongly in this random toy model), so compare the part
    # of the true change along the predicted one, row by row, as the study's calibration did
    ratio = (true_dh * dn).sum(1) / (dn * dn).sum(1)
    assert np.median(ratio) == pytest.approx(s, rel=0.3)
    assert np.linalg.norm(scaled_dh - true_dh) < 0.6 * np.linalg.norm(naive_dh - true_dh)


def test_logit_path_from_reader_matches_backend(be, tiny_dir):
    from weightprog.io.safetensors_np import SafetensorsReader

    L = be.spec.n_layers - 1
    p1 = LogitPath.from_backend(be, L)
    with SafetensorsReader(tiny_dir) as r:
        p2 = LogitPath.from_reader(r, be.spec, L)
        np.testing.assert_allclose(p1.E_rows([3, 7]), p2.E_rows([3, 7]), atol=1e-6)
        assert p1.scalar == pytest.approx(p2.scalar) and module_name(L, "down_proj")
