"""The console script end to end on the tiny model: tiny -> replay -> score -> controls -> gate -> export."""

import json
import subprocess
import sys

import numpy as np
import pytest

from weightprog.cli import main
from weightprog.edits import EditSet, ModuleEdit
from weightprog.testing.synthetic import synthetic_calls, write_call_set


def run(*args):
    return subprocess.run([sys.executable, "-m", "weightprog.cli", *map(str, args)], capture_output=True, text=True)


def test_help_and_templates(capsys):
    r = run("--help")
    assert r.returncode == 0 and "export-peft" in r.stdout
    main(["prereg"])
    assert "Pre-registration" in capsys.readouterr().out
    main(["spec"])
    assert "LIMITS" in capsys.readouterr().out


def test_pipeline(tmp_path, capsys):
    pytest.importorskip("mlx_lm")
    md = tmp_path / "model"
    main(["tiny", str(md), "--seed", "3"])
    capsys.readouterr()
    calls, files = synthetic_calls(6, seed=7, n_tasks=3)
    cs = write_call_set(tmp_path / "cs", calls, files)
    g = tmp_path / "gate"
    rng = np.random.default_rng(0)
    cand = g / "cand" / "edit"
    EditSet([ModuleEdit("model.language_model.layers.5.mlp.down_proj.weight", rng.standard_normal((64, 2)),
                        rng.standard_normal((2, 96)))]).save_dir(cand)
    main(["validate-edit", str(g / "cand"), "--model-dir", str(md)])
    assert json.loads(capsys.readouterr().out)[0]["rank"] == 2
    for k in (1, 2):
        main(["controls", str(g / "cand"), str(g / f"cand_rand{k}"), "--seed", str(100 + k)])
    capsys.readouterr()
    common = ["--backend", "mlx", "--model-dir", str(md), "--dtype", "float32", "--device", "cpu", "--calls",
              str(cs / "calls.jsonl"), "--keys", str(cs / "usable_keys.txt"), "--draws", "2"]
    main(["replay", "frozen", *common, "--out", str(g / "gen_base.jsonl")])
    for arm in ("cand", "cand_rand1", "cand_rand2"):
        main(["replay", "frozen", *common, "--edit", str(g / arm / "edit"), "--out", str(g / f"gen_{arm}.jsonl")])
    for arm in ("base", "cand", "cand_rand1", "cand_rand2"):
        main(["score", "frozen", str(g / f"gen_{arm}.jsonl"), "--calls", str(cs / "calls.jsonl"), "--files", str(cs / "files")])
    capsys.readouterr()
    main(["gate", "--gate-dir", str(g), "--keys", str(cs / "usable_keys.txt"), "--calls", str(cs / "calls.jsonl"),
          "--files", str(cs / "files"), "--m", "1", "--report", "rep.json", "cand"])
    out = capsys.readouterr().out
    assert "cand:" in out and "PRIMARY" in out
    rep = json.loads((g / "rep.json").read_text())
    assert rep["n_calls_x_draws"] == 12 and rep["n_tasks"] == 3
    main(["export-peft", str(cand), str(tmp_path / "adapter")])
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()
