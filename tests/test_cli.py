"""The console script end to end on the tiny model: tiny -> cases -> census -> eval (with controls, side effects) -> export."""

import json
import subprocess
import sys

import numpy as np
import pytest

from weightprog.cli import main
from weightprog.edits import EditSet, ModuleEdit
from weightprog.target import Case, save_cases


def run(*args):
    return subprocess.run([sys.executable, "-m", "weightprog.cli", *map(str, args)], capture_output=True, text=True)


def test_help():
    r = run("--help")
    assert r.returncode == 0 and "census" in r.stdout and "export-peft" in r.stdout


def test_pipeline(tmp_path, capsys):
    pytest.importorskip("mlx_lm")
    md = tmp_path / "model"
    main(["tiny", str(md), "--seed", "3"])
    cases = [Case(id=f"t{i}", prompt=f"def f{i}(x):\n    return", good=["x"], bad=["y"], max_new=5, group=f"g{i}")
             for i in range(4)]
    cases += [Case(id="k1", prompt="# note\n", role="keep", max_new=4),
              Case(id="h1", prompt="class B:\n", good=["self"], bad=["pass"], split="holdout", max_new=5)]
    save_cases(cases, tmp_path / "cases.jsonl")
    common = ["--backend", "mlx", "--model-dir", str(md), "--dtype", "float32", "--device", "cpu",
              "--cases", str(tmp_path / "cases.jsonl")]
    capsys.readouterr()
    main(["render", "--cases", str(tmp_path / "cases.jsonl"), "--id", "t0"])
    assert "def f0(x):" in capsys.readouterr().out
    main(["census", *common, "--split", "tune", "--out", str(tmp_path / "census.jsonl")])
    out = capsys.readouterr().out
    assert "t0" in out and "h1" not in out
    assert len((tmp_path / "census.jsonl").read_text().splitlines()) == 5 * 2
    rng = np.random.default_rng(0)
    edit = tmp_path / "edit"
    EditSet([ModuleEdit("model.language_model.layers.5.mlp.down_proj.weight", rng.standard_normal((64, 2)),
                        rng.standard_normal((2, 96)))]).save_dir(edit)
    main(["validate-edit", str(edit), "--model-dir", str(md)])
    assert json.loads(capsys.readouterr().out)[0]["rank"] == 2
    (tmp_path / "texts.json").write_text(json.dumps({"texts": ["def g(a):\n    return a\n", "Plain words here."]}))
    main(["eval", *common, "--split", "tune", "--edit", str(edit), "--controls", "2", "--side-effects",
          "--texts", str(tmp_path / "texts.json"), "--report", str(tmp_path / "rep.json")])
    out = capsys.readouterr().out
    assert "VERDICT" in out and "side effects" in out
    rep = json.loads((tmp_path / "rep.json").read_text())
    assert rep["controls"] == ["rand1", "rand2"] and rep["n_cases"] == 5
    assert set(rep["side_effects"]["by_arm"]) == {"edit", "rand1", "rand2"}
    # quick mode with a base-row cache: first call fills it, second reuses every case
    br = tmp_path / "base_rows.jsonl"
    for _ in range(2):
        main(["eval", *common, "--split", "tune", "--edit", str(edit), "--controls", "0", "--base-rows", str(br)])
    out, err = capsys.readouterr()
    assert "not measured: matched controls, side effects" in out and "5 cases reused, 0 generated" in err
    main(["controls", str(edit), str(tmp_path / "rand"), "--seed", "5"])
    main(["export-peft", str(edit), str(tmp_path / "adapter")])
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()
