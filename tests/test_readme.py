"""The README quickstart runs as written (with the model path moved to a temp dir)."""

import re
from pathlib import Path

import pytest


def test_quickstart(tmp_path, monkeypatch):
    pytest.importorskip("mlx_lm")
    s = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    code = re.search(r"## Quickstart.*?```python\n(.*?)```", s, re.S).group(1)
    code = code.replace('"/tmp/tiny_g4"', repr(str(tmp_path / "tiny")))
    ns = {}
    exec(compile(code, "README-quickstart", "exec"), ns)  # noqa: S102
    assert abs(ns["pred"] - (ns["new"] - ns["base"])).max() < 1e-3
