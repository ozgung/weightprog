import json

import numpy as np
import pytest

from weightprog.edits import EditError, EditSet, ModuleEdit, WeightSource, check_module
from weightprog.io.safetensors_np import SafetensorsReader, bf16_bits_to_f32, f32_to_bf16_bits, save_file
from weightprog.names import ModelSpec, canonical, file_stem, module_name, parse_module

DOWN5 = "model.language_model.layers.5.mlp.down_proj.weight"


@pytest.fixture(scope="module")
def spec(tiny_dir):
    return ModelSpec.from_config(tiny_dir)


@pytest.fixture(scope="module")
def weights(tiny_dir):
    return SafetensorsReader(tiny_dir / "model.safetensors")


def rand_edit(spec, name=DOWN5, r=2, seed=0, scale=0.1):
    out, inn = spec.module_shape(name)
    rng = np.random.default_rng(seed)
    return ModuleEdit(name, rng.standard_normal((out, r)) * scale, rng.standard_normal((r, inn)))


def test_names_and_spec(spec):
    assert spec.n_layers == 6 and spec.first_kv_shared == 4
    assert spec.kv_source == [0, 1, 2, 3, 3, 2]  # shared sliding layer 4 <- 3, shared global layer 5 <- 2
    assert spec.module_shape(DOWN5) == (64, 96)
    assert spec.module_shape(module_name(2, "q_proj")) == (4 * 32, 64)  # global head dim
    assert spec.module_shape(module_name(1, "k_proj")) == (2 * 16, 64)
    with pytest.raises(ValueError):
        spec.module_shape(module_name(4, "k_proj"))  # KV-shared layer
    # 4 non-shared layers x 7 + 2 shared x 5
    assert len(spec.editable_modules()) == 4 * 7 + 2 * 5
    assert canonical("language_model.model.layers.3.mlp.up_proj") == module_name(3, "up_proj")
    assert parse_module(DOWN5) == (5, "down_proj")
    assert file_stem(DOWN5) == "layers_5_mlp_down_proj"


def test_e4b_module_count():
    cfg = {"text_config": {"num_hidden_layers": 42, "num_kv_shared_layers": 18, "hidden_size": 2560,
                           "intermediate_size": 10240, "num_attention_heads": 8, "num_key_value_heads": 2,
                           "head_dim": 256, "global_head_dim": 512, "sliding_window": 512, "vocab_size": 262144,
                           "layer_types": ["full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(42)]}}
    s = ModelSpec.from_config(cfg)
    assert len(s.editable_modules()) == 258
    assert s.kv_source[24] == 22 and s.kv_source[29] == 23 and s.kv_source[41] == 23


def test_npz_roundtrip_and_original_format(tmp_path, spec, weights):
    e = rand_edit(spec)
    es = EditSet([e])
    es.save_dir(tmp_path / "edit")
    assert [f.name for f in (tmp_path / "edit").glob("*.npz")] == ["layers_5_mlp_down_proj.npz"]
    back = EditSet.load(tmp_path)  # run folder -> its edit/
    np.testing.assert_array_equal(back[DOWN5].B, e.B)
    np.testing.assert_array_equal(back[DOWN5].A, e.A)
    # a file written exactly as the original kernel did (np.savez, meta as a JSON string)
    meta = {"module": DOWN5, "rank": 2, "note": "orig"}
    np.savez(tmp_path / "orig.npz", module=np.array(DOWN5), B=e.B, A=e.A, meta=np.array(json.dumps(meta)))
    o = ModuleEdit.from_npz(tmp_path / "orig.npz")
    assert o.meta["note"] == "orig" and o.rank == 2
    snap = EditSet.snapshot(tmp_path / "edit")
    assert set(snap) == {"layers_5_mlp_down_proj.npz"} and len(next(iter(snap.values()))) == 64


def test_validation(spec, weights):
    rec = check_module(rand_edit(spec), spec, weights)
    W = weights.get(DOWN5)
    assert rec["weight_fro"] == pytest.approx(float(np.linalg.norm(W)))
    assert rec["rel_change"] == pytest.approx(rec["delta_fro"] / rec["weight_fro"])
    with pytest.raises(EditError, match="not editable"):
        check_module(ModuleEdit("model.language_model.norm.weight", np.ones((64, 1)), np.ones((1, 64))), spec, weights)
    with pytest.raises(EditError, match="not in this model"):
        check_module(ModuleEdit(module_name(4, "k_proj"), np.ones((32, 1)), np.ones((1, 64))), spec, weights)
    with pytest.raises(EditError, match="shapes"):
        check_module(ModuleEdit(DOWN5, np.ones((63, 1)), np.ones((1, 96))), spec, weights)
    with pytest.raises(EditError, match="rank"):
        check_module(ModuleEdit(DOWN5, np.ones((64, 129)), np.ones((129, 96))), spec, weights)
    bad = rand_edit(spec)
    bad.B[0, 0] = np.nan
    with pytest.raises(EditError, match="non-finite"):
        check_module(bad, spec, weights)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)  # rank 64 <= 128: B @ A = 0.1 W exactly
    with pytest.raises(EditError, match="rescale"):
        check_module(ModuleEdit(DOWN5, 0.1 * U * S, Vt), spec, weights)


def test_editset_container(spec, weights):
    es = EditSet([rand_edit(spec), rand_edit(spec, module_name(1, "q_proj"), r=3, seed=2)])
    assert es.modules() == [module_name(1, "q_proj"), DOWN5]  # file-name order
    es.add(rand_edit(spec, seed=5), replace=True)  # replaces
    assert len(es) == 2
    half = es.scaled(0.5)
    np.testing.assert_allclose(half[DOWN5].delta(), 0.5 * es[DOWN5].delta(), rtol=1e-6)
    n = es.norms(weights)
    assert {r["module"] for r in n} == set(es.modules())
    ws = WeightSource({DOWN5: weights.get(DOWN5)})
    assert DOWN5 in ws and ws.shape(DOWN5) == (64, 96)


def test_safetensors_bf16_roundtrip(tmp_path):
    x = np.random.default_rng(0).standard_normal((5, 7)).astype(np.float32)
    bits = f32_to_bf16_bits(x)
    y = bf16_bits_to_f32(bits)
    assert np.abs(x - y).max() <= np.abs(x).max() * 2 ** -8
    np.testing.assert_array_equal(bf16_bits_to_f32(f32_to_bf16_bits(y)), y)  # bf16 values are fixed points
    save_file({"a": x, "b": np.arange(3, dtype=np.int64)}, tmp_path / "t.safetensors", dtypes={"a": "BF16"})
    with SafetensorsReader(tmp_path / "t.safetensors") as r:
        assert r.dtype("a") == "BF16" and r.shape("a") == (5, 7)
        np.testing.assert_array_equal(r.get("a"), y)
        np.testing.assert_array_equal(r.get("b"), np.arange(3))


def test_safetensors_reader_matches_torch(tiny_dir_bf16):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    ref = st.load_file(str(tiny_dir_bf16 / "model.safetensors"))
    with SafetensorsReader(tiny_dir_bf16) as r:
        for k in list(ref)[:10]:
            np.testing.assert_array_equal(r.get(k), ref[k].to(torch.float32).numpy())
