from pathlib import Path

import numpy as np
import pytest

from conftest import env_path
from weightprog.controls import control_seeds, matched_control, write_control
from weightprog.edits import EditSet, ModuleEdit


def src_edit():
    rng = np.random.default_rng(3)
    B = rng.standard_normal((64, 2)).astype(np.float32)
    B[10:20] = 0
    return EditSet([ModuleEdit("model.language_model.layers.5.mlp.down_proj.weight", B, rng.standard_normal((2, 96))),
                    ModuleEdit("model.language_model.layers.1.self_attn.q_proj.weight",
                               rng.standard_normal((64, 3)), rng.standard_normal((3, 64)))])


def test_matched_norms_and_determinism():
    s = src_edit()
    c1, c1b, c2 = matched_control(s, 7), matched_control(s, 7), matched_control(s, 8)
    for e in s:
        a, b = c1[e.module], c2[e.module]
        assert a.B.shape == e.B.shape and a.A.shape == e.A.shape
        assert np.linalg.norm(a.delta()) == pytest.approx(np.linalg.norm(e.delta()), rel=1e-5)
        np.testing.assert_array_equal(a.B, c1b[e.module].B)
        assert not np.allclose(a.B, b.B)


def test_support_matched(tmp_path):
    s = src_edit()
    c = matched_control(s, 1, support_matched=True)
    assert np.all(c["model.language_model.layers.5.mlp.down_proj.weight"].B[10:20] == 0)
    s.save_dir(tmp_path / "src" / "edit")
    rec = write_control(tmp_path / "src", tmp_path / "ctrl", seed=1)
    assert len(rec) == 2 and (tmp_path / "ctrl" / "control.json").exists()
    assert control_seeds(20261104, 2) == {(1, 1): 20261115, (1, 2): 20261116, (2, 1): 20261125, (2, 2): 20261126}


@pytest.mark.regression
def test_reproduces_study_controls():
    """With WEIGHTPROG_GATE_DIR = the study's gate_r2 folder: controls rebuilt from the frozen candidate edits with
    the registered seeds equal the stored control files bit for bit."""
    import json

    g = env_path("WEIGHTPROG_GATE_DIR")
    if not g:
        pytest.skip("WEIGHTPROG_GATE_DIR not set")
    checked = 0
    for cand in ("opus_r2", "fable_r2"):
        for k in (1, 2):
            cdir = Path(g) / f"{cand}_rand{k}"
            if not (cdir / "control.json").exists():
                continue
            seed = json.loads((cdir / "control.json").read_text())[0]["seed"]
            ctrl = matched_control(EditSet.load_dir(Path(g) / cand / "edit"), seed)
            stored = EditSet.load_dir(cdir / "edit")
            for e in stored:
                np.testing.assert_array_equal(ctrl[e.module].B, e.B)
                np.testing.assert_array_equal(ctrl[e.module].A, e.A)
            checked += 1
    assert checked
