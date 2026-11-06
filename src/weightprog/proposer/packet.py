"""Proposer packets and briefs.

    text = fill_brief(brief_template(), {"RUN_DIR": ..., "BUDGET": "10", ...})   # every {UPPER_CASE} must be filled
    build_packet(packet_dir, brief=text, call_set=dir_with_calls_jsonl_and_files, model_dir=..., dossier=path,
                 prior=None, limits=LIMITS)

The packet is the proposer's read-only working directory: brief.md, dossier.md, server_spec.md, search/ (calls.jsonl,
optional calls_raw.jsonl, MANIFEST.txt, files/), kernel.py + safetensors_np.py + kernel_config.json (a standalone
weight kernel pointing at the model directory), optional prior/ (earlier runs, for an informed arm) and a writable
scratch/. Everything but scratch/ is made read-only.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
from importlib.resources import files as _files
from pathlib import Path

PLACEHOLDER = re.compile(r"\{[A-Z][A-Z0-9_]*\}")


def brief_template() -> str:
    return (_files("weightprog.proposer") / "brief_template.md").read_text()


def fill_brief(template: str, values: dict[str, str], allow_missing: bool = False) -> str:
    out = template
    for k, v in values.items():
        out = out.replace("{" + k + "}", str(v))
    left = sorted(set(PLACEHOLDER.findall(out)))
    if left and not allow_missing:
        raise ValueError(f"unfilled placeholders: {left}")
    return out


def _readonly(p: Path) -> None:
    for root, dirs, fs in os.walk(p):
        for f in fs:
            fp = Path(root) / f
            fp.chmod(fp.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _writable(p: Path) -> None:
    if not p.exists():
        return
    for root, dirs, fs in os.walk(p):
        Path(root).chmod(Path(root).stat().st_mode | stat.S_IWUSR)
        for f in fs:
            fp = Path(root) / f
            fp.chmod(fp.stat().st_mode | stat.S_IWUSR)


def build_packet(packet_dir: str | Path, brief: str, call_set: str | Path, model_dir: str | Path,
                 dossier: str | Path | None = None, prior: str | Path | None = None, limits: dict | None = None) -> Path:
    from .server import spec_text

    pk = Path(packet_dir)
    _writable(pk)
    (pk / "scratch").mkdir(parents=True, exist_ok=True)
    cs = Path(call_set)
    search = pk / "search"
    if search.exists():
        shutil.rmtree(search)
    search.mkdir()
    for name in ("calls.jsonl", "calls_raw.jsonl", "MANIFEST.txt"):
        if (cs / name).exists():
            shutil.copy2(cs / name, search / name)
    shutil.copytree(cs / "files", search / "files")
    (pk / "brief.md").write_text(brief)
    (pk / "dossier.md").write_text(Path(dossier).read_text() if dossier else "# Dossier\n\n(no prior facts)\n")
    (pk / "server_spec.md").write_text(spec_text(limits))
    src = Path(__file__).resolve().parent
    shutil.copy2(src / "kernel.py", pk / "kernel.py")
    shutil.copy2(src.parent / "io" / "safetensors_np.py", pk / "safetensors_np.py")
    (pk / "kernel_config.json").write_text(json.dumps({"model_dir": str(Path(model_dir).resolve())}))
    if prior:
        if (pk / "prior").exists():
            shutil.rmtree(pk / "prior")
        shutil.copytree(prior, pk / "prior")
    for p in pk.iterdir():
        if p.name != "scratch":
            if p.is_dir():
                _readonly(p)
            else:
                p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    return pk
