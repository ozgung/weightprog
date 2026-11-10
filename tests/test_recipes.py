"""Recipe code runs as written on the tiny model (API names and shapes stay in sync with the docs); every
`from weightprog... import X` in the docs and skills resolves."""

import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted([*ROOT.glob("docs/**/*.md"), *ROOT.glob("skills/**/*.md"), ROOT / "AGENTS.md", ROOT / "README.md"])


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_doc_imports_resolve(path):
    text = path.read_text()
    found = re.findall(r"from (weightprog[\w\.]*) import \(([\w, \n]+)\)", text)
    found += re.findall(r"from (weightprog[\w\.]*) import ([\w, ]+)$", text, re.M)
    for mod, names in found:
        m = importlib.import_module(mod)
        for n in (x.strip() for x in names.replace("\n", " ").split(",")):
            if n:
                assert hasattr(m, n), f"{path.name}: {mod}.{n} does not exist"


def _code(name):
    s = (ROOT / "docs/recipes" / name).read_text()
    return re.search(r"## Code\n\n```python\n(.*?)```", s, re.S).group(1)


@pytest.mark.parametrize("name", ["context-token-choice.md", "global-preference.md", "remove-a-behaviour.md"])
def test_recipe_runs_on_tiny(name, tiny_dir, tmp_path, monkeypatch):
    pytest.importorskip("mlx_lm")
    code = _code(name).replace('"/path/to/gemma-4-e4b-it"', repr(str(tiny_dir)))
    code = code.replace("open_model(MODEL_DIR)", 'open_model(MODEL_DIR, backend="mlx", dtype="float32", device="cpu")')
    monkeypatch.chdir(tmp_path)
    exec(compile(code, name, "exec"), {})  # noqa: S102
