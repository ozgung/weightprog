"""MLX vs PyTorch on the same tiny checkpoint, float32 on CPU."""

import numpy as np
import pytest

from weightprog.backends import Force, Generate
from weightprog.edits import EditSet, ModuleEdit
from weightprog.names import module_name

ALL = {"resid", "resid_pre", "resid_mid", "mlp", "mlp_out", "pli", "attn", "logits"}


def test_native_logprobs_agree(mlx_be, torch_be, sample_ids):
    pos = list(range(len(sample_ids)))
    np.testing.assert_allclose(mlx_be.logprobs(sample_ids, pos), torch_be.logprobs(sample_ids, pos), atol=1e-3)


def test_captures_agree(mlx_be, torch_be, sample_ids):
    pos = [0, 7, 8, 9, 30, len(sample_ids) - 1]
    a = mlx_be.capture(sample_ids, pos, layers=range(6), what=ALL, attn_layers=[2, 4, 5])
    b = torch_be.capture(sample_ids, pos, layers=range(6), what=ALL, attn_layers=[2, 4, 5])
    for k in ALL:
        assert a[k].shape == b[k].shape, k
        scale = max(1.0, float(np.abs(a[k]).max()))
        np.testing.assert_allclose(a[k], b[k], atol=2e-4 * scale, err_msg=k)


def test_greedy_generation_identical(mlx_be, torch_be, sample_ids):
    q = mlx_be.tokenizer.token_to_id('<|"|>')
    ga = mlx_be.generate(sample_ids, 40, draws=1, stop_ids=[q])[0]
    gb = torch_be.generate(sample_ids, 40, draws=1, stop_ids=[q])[0]
    assert ga.tokens == gb.tokens and ga.closed == gb.closed


def test_segments_identical_under_edit(mlx_be, torch_be, sample_ids):
    rng = np.random.default_rng(5)
    es = EditSet([ModuleEdit(module_name(5, "down_proj"), rng.standard_normal((64, 2)), rng.standard_normal((2, 96))),
                  ModuleEdit(module_name(2, "o_proj"), rng.standard_normal((64, 1)) * 0.3, rng.standard_normal((1, 128)))])
    q = mlx_be.tokenizer.token_to_id('<|"|>')
    segs = [Generate(15, [q]), Force([q] + mlx_be.tokenizer.encode(",b:") + [q]), Generate(15, [q])]
    outs = {}
    for be in (mlx_be, torch_be):
        with be.edits(es):
            outs[be.name] = be.decode_segments(sample_ids, segs, draws=1, abort_on_unclosed=False)[0]
    assert [s.tokens for s in outs["mlx"]] == [s.tokens for s in outs["torch"]]
    base = mlx_be.decode_segments(sample_ids, segs, draws=1, abort_on_unclosed=False)[0]
    assert [s.tokens for s in base] != [s.tokens for s in outs["mlx"]]


def test_bf16_loads_in_both(tiny_dir_bf16, sample_ids):
    """bf16 checkpoint, bf16 compute: both load strictly and agree loosely (bf16 rounding differs by backend)."""
    from conftest import BACKENDS
    from weightprog.backends import get_backend

    if not {"mlx", "torch"} <= set(BACKENDS):
        pytest.skip("needs both backends")
    a = get_backend("mlx", tiny_dir_bf16, dtype="bfloat16", device="cpu")
    b = get_backend("torch", tiny_dir_bf16, dtype="bfloat16", device="cpu")
    assert a.dtype == "mlx.core.bfloat16" or "bfloat16" in a.dtype
    pos = list(range(len(sample_ids) - 5, len(sample_ids)))
    pa, pb = np.exp(a.logprobs(sample_ids, pos)), np.exp(b.logprobs(sample_ids, pos))
    assert np.abs(pa - pb).max() < 0.05
    np.testing.assert_array_equal(a.read_weight(module_name(0, "q_proj")), b.read_weight(module_name(0, "q_proj")))
