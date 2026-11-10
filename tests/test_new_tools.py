"""kv_update / second_moment, cv_gate, residual_write_direction, decision_from_text, layer_sweep, EditSet.merge,
bf16 realised fraction, read_rows and shared tail weights."""

import numpy as np
import pytest

from weightprog.construct import (TailState, cv_gate, kv_update, predict_logit_shift, residual_write_direction,
                                  second_moment)
from weightprog.construct.linear import rms_norm
from weightprog.edits import EditError, EditSet, ModuleEdit, bf16_round, realised_fraction
from weightprog.names import module_name
from weightprog.probe import collect_tail, decision_from_text, layer_sweep


def test_kv_update_hits_keys_and_stays_quiet():
    rng = np.random.default_rng(0)
    d, H = 40, 12
    common = rng.standard_normal((500, d)) @ np.diag(np.r_[np.full(20, 3.0), np.full(20, 0.05)])  # text uses dims 0-19
    C = common.T @ common / len(common)
    K = np.zeros((3, d)); K[:, 20:23] = np.eye(3) * 2.0     # keys in directions ordinary text barely uses
    R = rng.standard_normal((3, H))
    e = kv_update(0, K, R, C, lam=1.0, eps=1e-6)
    dW = e.delta()
    np.testing.assert_allclose(dW @ K.T, R.T, atol=0.05)    # hits the keys
    quiet = np.linalg.norm(common @ dW.T, axis=1).mean() / np.linalg.norm(R, axis=1).mean()
    assert quiet < 0.05 and e.rank == 3 and e.module == module_name(0, "down_proj")
    with pytest.raises(ValueError):
        kv_update(0, np.zeros((129, d)), np.zeros((129, H)), C)


def test_second_moment_and_cv_gate(be):
    L = be.spec.n_layers - 1
    C, n = second_moment(be, ["def f(x):\n    return x", "plain words"], L)
    assert C.shape == (be.spec.intermediate_size(L), be.spec.intermediate_size(L)) if hasattr(be.spec, "intermediate_size") else C.shape[0] == C.shape[1]
    assert n > 10 and np.allclose(C, C.T)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 10)); y = (X[:, 0] > 0).astype(float)
    cv = cv_gate(X, y, lams=(0.1, 10.0))
    assert set(cv) == {0.1, 10.0} and cv[0.1]["auc"] > 0.8 and len(cv[0.1]["oof_gate"]) == 400


def test_residual_write_direction(be):
    L = be.spec.n_layers - 2
    r = np.random.default_rng(2).standard_normal(be.read_weight(module_name(L, "down_proj")).shape[0])
    u = residual_write_direction(be, L, r, "mlp")
    w = be.read_weight(f"model.language_model.layers.{L}.post_feedforward_layernorm.weight")
    contrib = rms_norm(u[None], w)[0]                       # what the residual receives for module output u
    cos = contrib @ r / (np.linalg.norm(contrib) * np.linalg.norm(r))
    assert cos > 0.999


def test_decision_from_text(be):
    tok = be.tokenizer
    d = decision_from_text(tok, "x = obj.", "dict() + 1", r"(dict)\(", group=1)
    assert d is not None and tok.decode(d.prefix_ids) == "x = obj." and tok.decode([d.bad_id]) == "d"
    assert decision_from_text(tok, "x", "nothing here", r"dict") is None


def test_layer_sweep(be):
    tok = be.tokenizer
    pos = [tok.encode(f"def f{i}(x): return x") for i in range(6)]
    neg = [tok.encode(f"# comment {i} about prose") for i in range(6)]
    out = layer_sweep(be, pos, neg, layers=[1, be.spec.n_layers - 1])
    assert [o["layer"] for o in out] == [1, be.spec.n_layers - 1] and all(0 <= o["auc"] <= 1 for o in out)


def test_merge_and_duplicates():
    rng = np.random.default_rng(3)
    m = module_name(1, "down_proj")
    a, b = ModuleEdit(m, rng.standard_normal((8, 1)), rng.standard_normal((1, 5))), ModuleEdit(m, rng.standard_normal((8, 2)), rng.standard_normal((2, 5)))
    es = EditSet([a])
    with pytest.raises(EditError):
        es.add(b)
    es.merge(b)
    assert es[m].rank == 3
    np.testing.assert_allclose(es[m].delta(), a.delta() + b.delta(), atol=1e-5)


def test_bf16_round_and_realised_fraction():
    x = np.array([1.0, 1.00390625, 1.0078125, -3.14159], dtype=np.float32)
    r = bf16_round(x)
    assert r[0] == 1.0 and r[2] == np.float32(1.0078125) and abs(r[3] + 3.140625) < 1e-6
    W = bf16_round(np.random.default_rng(4).standard_normal((64, 32)).astype(np.float32))
    big = ModuleEdit("model.language_model.layers.0.mlp.down_proj.weight", np.ones((64, 1)) * 0.5, np.ones((1, 32)))
    tiny = big.scaled(1e-5)
    f = realised_fraction(EditSet([big]), lambda n: W)
    g = realised_fraction(EditSet([tiny]), lambda n: W)
    assert f[big.module] > 0.95 and g[tiny.module] < 0.5
    assert realised_fraction(EditSet([tiny]), lambda n: W, dtype="float32")[tiny.module] > 0.99


def test_read_rows_and_shared_tail_weights(be):
    name = "model.language_model.embed_tokens.weight"
    np.testing.assert_allclose(be.read_rows(name, [3, 7, 3]), be.read_weight(name)[[3, 7, 3]], atol=0)
    tok = be.tokenizer
    L = be.spec.n_layers - 1
    a, b = tok.encode("def f(x):\n    return x"), tok.encode("class Q:\n    pass")
    st = collect_tail(be, [(a, [3, 5]), (b, [4])], L)
    one = TailState.from_backend(be, b, [4], L)
    np.testing.assert_allclose(st.act[2], one.act[0], atol=1e-6)
    e = ModuleEdit(module_name(L, "down_proj"), np.ones((st.W_down.shape[0], 1)) * 0.1, np.ones((1, st.W_down.shape[1])) * 0.1)
    p_all = predict_logit_shift(st, e, [3, 4])
    p_one = predict_logit_shift(one, e, [3, 4])
    np.testing.assert_allclose(p_all[2], p_one[0], atol=1e-6)


def test_unit_patch_rewires_only_chosen_units(be):
    from weightprog.construct import mlp_input, quiet_units, unit_patch
    from weightprog.construct.linear import gelu_tanh
    tok = be.tokenizer
    L = be.spec.n_layers - 1
    ids = tok.encode("def f(x):\n    return x + 1")
    pos = list(range(3, len(ids)))
    base = be.capture(ids, pos, layers=[L], what={"mlp", "resid_mid"})
    Wd = be.read_weight(module_name(L, "down_proj"))
    units = quiet_units(Wd, np.sqrt((base["mlp"][0] ** 2).mean(0)), 3)
    rng = np.random.default_rng(5)
    H = Wd.shape[0]
    G, U, D = rng.standard_normal((3, H)) * 0.1, rng.standard_normal((3, H)) * 0.1, rng.standard_normal((H, 3)) * 0.1
    es = unit_patch(be.read_weight, L, units, G, U, D)
    assert len(es) == 3 and all(e.rank == 3 for e in es)
    with be.edits(es):
        new = be.capture(ids, pos, layers=[L], what={"mlp"})["mlp"][0]
    x = mlp_input(be, base["resid_mid"][0], L)
    want = gelu_tanh(x @ G.T) * (x @ U.T)
    np.testing.assert_allclose(new[:, units], want, atol=2e-4)
    others = np.setdiff1d(np.arange(new.shape[1]), units)
    np.testing.assert_allclose(new[:, others], base["mlp"][0][:, others], atol=1e-5)
    dn = module_name(L, "down_proj")
    np.testing.assert_allclose((be.read_weight(dn) + es[dn].delta())[:, units], D, atol=1e-5)


def test_null_space_projection_preserves_keys():
    from weightprog.construct import kv_update, null_space_basis, project_edit
    rng = np.random.default_rng(6)
    d, H = 30, 8
    keep_keys = rng.standard_normal((200, 10)) @ rng.standard_normal((10, d))   # preserved inputs span 10 dims
    C = keep_keys.T @ keep_keys / 200
    U = null_space_basis(C, rel_tol=1e-6)
    assert U.shape == (d, 20)
    e = ModuleEdit(module_name(0, "down_proj"), rng.standard_normal((H, 2)), rng.standard_normal((2, d)))
    p = project_edit(e, U)
    assert np.abs(keep_keys @ p.delta().T).max() < 1e-4 and np.abs(keep_keys @ e.delta().T).max() > 1   # float32 storage
    K = rng.standard_normal((3, d)); R = rng.standard_normal((3, H))
    kv = kv_update(0, K, R, C, lam=1.0, eps=1e-6, null_basis=U)
    assert np.abs(keep_keys @ kv.delta().T).max() < 1e-4
    np.testing.assert_allclose(kv.delta() @ (K @ U @ U.T).T, R.T, atol=0.05)   # hits the keys' null-space part


def test_patch_effect(be):
    from weightprog.probe import patch_effect, patch_sweep
    tok = be.tokenizer
    clean, corrupt = tok.encode("say: alpha beta"), tok.encode("say: gamma beta")
    assert len(clean) == len(corrupt)
    g, b = tok.encode("a")[0], tok.encode("x")[0]
    last = len(clean) - 1
    full = patch_effect(be, clean, corrupt, be.spec.n_layers - 1, list(range(len(clean))), g, b)
    assert full["restored"] == pytest.approx(1.0, abs=1e-3)       # patching everything at the last layer restores all
    lp = be.logprobs(corrupt, [last])[0]
    assert full["corrupt"] == pytest.approx(float(lp[g] - lp[b]), abs=1e-3)
    sw = patch_sweep(be, clean, corrupt, g, b, layers=[0, be.spec.n_layers - 1])
    assert {r["where"] for r in sw} == {"differing tokens", "last position"} and len(sw) == 4
