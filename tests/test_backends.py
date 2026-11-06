"""Per-backend behaviour on the tiny model (each test runs for MLX and for PyTorch)."""

import numpy as np
import pytest

from weightprog.backends import Force, Generate, Sampling
from weightprog.edits import EditSet, ModuleEdit
from weightprog.names import module_name

DOWN5 = module_name(5, "down_proj")


def logsoftmax(x):
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def an_edit(be, scale=0.5):
    rng = np.random.default_rng(11)
    es = EditSet([ModuleEdit(DOWN5, rng.standard_normal((64, 2)) * scale, rng.standard_normal((2, 96))),
                  ModuleEdit(module_name(1, "v_proj"), rng.standard_normal((32, 1)) * scale, rng.standard_normal((1, 64)))])
    return es


def test_weights_and_names(be):
    names = be.weight_names()
    assert DOWN5 in names and module_name(4, "k_proj") not in names
    W = be.read_weight(DOWN5)
    assert W.shape == (64, 96) and W.dtype == np.float32
    s = be.read_weight("model.language_model.layers.3.layer_scalar")
    assert s.shape == (1,) and 0.4 <= float(s[0]) <= 1.0


def test_edit_apply_restore_exact(be, sample_ids):
    es = an_edit(be)
    W0 = {m: be.read_weight(m) for m in es.modules()}
    base = be.logprobs(sample_ids, [len(sample_ids) - 1])
    with be.edits(es):
        for e in es:
            np.testing.assert_allclose(be.read_weight(e.module), W0[e.module] + e.delta(), rtol=1e-6, atol=1e-6)
        edited = be.logprobs(sample_ids, [len(sample_ids) - 1])
        with be.edits(EditSet([es[DOWN5].scaled(-1.0)])):  # nesting: inner edit undone first
            np.testing.assert_allclose(be.read_weight(DOWN5), W0[DOWN5], atol=1e-5)
        np.testing.assert_allclose(be.read_weight(DOWN5), W0[DOWN5] + es[DOWN5].delta(), rtol=1e-6, atol=1e-6)
    for m, w in W0.items():
        np.testing.assert_array_equal(be.read_weight(m), w)  # bitwise restore
    assert np.abs(edited - base).max() > 1e-3
    np.testing.assert_array_equal(be.logprobs(sample_ids, [len(sample_ids) - 1]), base)


def test_edit_failure_restores(be):
    W0 = be.read_weight(DOWN5)
    bad = EditSet([ModuleEdit(DOWN5, np.ones((64, 1)), np.ones((1, 96))),
                   ModuleEdit(module_name(3, "up_proj"), np.ones((95, 1)), np.ones((1, 64)))])  # wrong shape
    with pytest.raises(ValueError):
        with be.edits(bad):
            pass
    np.testing.assert_array_equal(be.read_weight(DOWN5), W0)


def test_capture_matches_native(be, sample_ids):
    L = len(sample_ids)
    assert L > 2 * be.spec.sliding_window
    pos = [0, 5, 17, L - 2, L - 1]
    cap = be.capture(sample_ids, pos, layers=range(6), what={"resid", "mlp", "logits", "attn", "resid_mid", "mlp_out", "pli", "resid_pre"},
                     attn_layers=[3, 4, 5])
    nat = be.logprobs(sample_ids, pos)
    np.testing.assert_allclose(logsoftmax(cap["logits"].astype(np.float64)), nat, atol=2e-4)
    assert cap["resid"].shape == (6, 5, 64) and cap["mlp"].shape == (6, 5, 96) and cap["pli"].shape == (6, 5, 8)
    assert cap["attn"].shape == (3, 4, 5, L)
    A = cap["attn"]
    np.testing.assert_allclose(A.sum(-1), 1.0, atol=1e-5)
    W = be.spec.sliding_window
    for li, layer in enumerate([3, 4, 5]):
        for pi, p in enumerate(pos):
            assert np.all(A[li, :, pi, p + 1:] == 0)  # causal
            if be.spec.is_sliding(layer):
                assert np.all(A[li, :, pi, : max(0, p - W + 1)] < 1e-20)  # outside the window
                assert np.all(A[li, :, pi, max(0, p - W + 1): p + 1].sum(-1) > 0.999)
    # residual bookkeeping: resid_pre of layer i+1 == resid of layer i
    np.testing.assert_allclose(cap["resid_pre"][1:], cap["resid"][:-1], atol=1e-5)
    # mlp_out == down_proj(mlp)
    Wd = be.read_weight(DOWN5)
    np.testing.assert_allclose(cap["mlp_out"][5], cap["mlp"][5] @ Wd.T, atol=1e-4)


def test_logprobs_positions(be, sample_ids):
    a = be.logprobs(sample_ids, list(range(10, 20)))
    b = be.logprobs(sample_ids, [19, 12])
    np.testing.assert_allclose(b, a[[9, 2]], atol=5e-4)  # different prefill split, same math
    with pytest.raises(ValueError):
        be.logprobs(sample_ids, [len(sample_ids)])


def test_greedy_is_argmax_and_seeds_reproduce(be, sample_ids):
    q = be.tokenizer.token_to_id('<|"|>')
    g = be.generate(sample_ids, 24, draws=3, seeds=[1, 2, 2], stop_ids=[q])
    assert g[1].tokens != [] and g[2].tokens == g[1].tokens  # same seed, same sample
    seq = list(sample_ids)
    toks = g[0].tokens
    lp = be.logprobs(seq + toks, list(range(len(seq) - 1, len(seq) - 1 + len(toks))))
    assert list(lp.argmax(-1)) == toks


def test_segments_cache_consistency(be, sample_ids):
    """Generate / Force / Generate on one cache: every greedy token is the argmax of a fresh forward over the whole
    sequence (so the KV cache, sliding window and KV sharing behave across segments)."""
    stop = be.tokenizer.token_to_id("x")
    forced = be.tokenizer.encode(",old:") + [be.tokenizer.token_to_id('<|"|>')]
    res = be.decode_segments(sample_ids, [Generate(12, [stop]), Force(forced), Generate(12, [stop])], draws=1,
                             abort_on_unclosed=False)[0]
    g1, f, g2 = res
    assert f.tokens == forced
    full = list(sample_ids) + g1.tokens + forced + g2.tokens
    start = len(sample_ids) - 1
    lp1 = be.logprobs(full, list(range(start, start + len(g1.tokens))))
    assert list(lp1.argmax(-1)) == g1.tokens
    s2 = len(sample_ids) + len(g1.tokens) + len(forced) - 1
    if g2.tokens:
        lp2 = be.logprobs(full, list(range(s2, s2 + len(g2.tokens))))
        assert list(lp2.argmax(-1)) == g2.tokens


def test_unclosed_aborts(be, sample_ids):
    res = be.decode_segments(sample_ids, [Generate(3, []), Force([5]), Generate(3, [])], draws=2, seeds=[0, 1])
    for r in res:
        assert r[0].closed == 0 and len(r[0].tokens) == 3 and r[1].skipped and r[2].skipped
    res = be.decode_segments(sample_ids, [Generate(3, []), Force([5]), Generate(3, [])], draws=1, abort_on_unclosed=False)
    assert len(res[0][2].tokens) == 3


def test_sampling_respects_top_k(be, sample_ids):
    lp = be.logprobs(sample_ids, [len(sample_ids) - 1])[0]
    top2 = set(np.argsort(-lp)[:2].tolist())
    res = be.decode_segments(sample_ids, [Generate(1, [])], draws=13, seeds=list(range(13)),
                             sampling=Sampling(temperature=5.0, top_p=1.0, top_k=2))
    assert {r[0].tokens[0] for r in res[1:]} <= top2


def test_mlx_loop_equals_generate_step(mlx_be, sample_ids):
    """The explicit decode loop reproduces mlx_lm's generate_step (the study's frozen-prefix generator), greedy and
    sampled with the same seed, so cached generations from the study's server stay comparable."""
    import copy

    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    q = mlx_be.tokenizer.token_to_id('<|"|>')
    ours = mlx_be.generate(sample_ids, 30, draws=3, seeds=[3, 4, 5], stop_ids=[q])
    cache = mlx_be._prefill(sample_ids[:-1])
    for d in range(3):
        sampler = make_sampler(temp=0.0) if d == 0 else make_sampler(temp=0.6, top_p=0.95, top_k=64)
        mx.random.seed([3, 4, 5][d])
        gen = []
        for tid, _ in generate_step(mx.array(sample_ids[-1:]), mlx_be.model, max_tokens=30, sampler=sampler,
                                    prompt_cache=copy.deepcopy(cache)):
            if int(tid) == q:
                break
            gen.append(int(tid))
        assert gen == ours[d].tokens, d
