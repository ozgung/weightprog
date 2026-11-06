"""Regression: the library gate reproduces the study's registered round-2 report exactly.

Run with the study's folders (read-only):

    WEIGHTPROG_GATE_DIR=.../outputs/weight_edit/claude/gate_r2 \\
    WEIGHTPROG_REPLAY_DIR=.../outputs/weight_edit/claude/replay_ood  pytest tests/test_gate_regression.py

Skipped when the variables are unset. Nothing is written to those folders.
"""

import json
from pathlib import Path

import pytest

from conftest import env_path
from weightprog.gate import GateSpec, amend, analyze, fullcall_summary, load_arm, universe
from weightprog.gate.io import read_jsonl_unique
from weightprog.scoring import escape_artefact

pytestmark = pytest.mark.regression


@pytest.fixture(scope="module")
def study():
    g, r = env_path("WEIGHTPROG_GATE_DIR"), env_path("WEIGHTPROG_REPLAY_DIR")
    if not (g and r):
        pytest.skip("WEIGHTPROG_GATE_DIR / WEIGHTPROG_REPLAY_DIR not set")
    g, r = Path(g), Path(r)
    keys = universe(r / "usable_keys.txt", (0, 1))
    want = {k for k, _ in keys}
    calls = {}
    with (r / "calls.jsonl").open() as f:
        for line in f:
            c = json.loads(line)
            calls[c["key"]] = {"repo": c["repo"], "file_sha": c["file_sha"]}  # drop prompts (large)
    files = {}

    def file_text(key):
        sha = calls[key]["file_sha"]
        if sha not in files:
            files[sha] = (r / "files" / f"{sha}.txt").read_text()
        return files[sha]

    metrics = {"escape_artefact": lambda gen: escape_artefact(gen["gen"], file_text(gen["key"]))}
    cluster_of = {k.split("|")[1]: calls[k]["repo"] for k in want}
    return {"g": g, "keys": keys, "metrics": metrics, "cluster_of": cluster_of}


def test_registered_report_reproduced_exactly(study):
    g, keys = study["g"], study["keys"]
    expected = json.loads((g / "gate_r2_report.json").read_text())
    cands = list(expected["candidates"])
    base = load_arm(g, "base", keys, False, study["metrics"])
    arms = {c: load_arm(g, c, keys, True, study["metrics"]) for c in cands}
    ctrls = {c: {x: load_arm(g, f"{c}_{x}", keys, True, study["metrics"]) for x in ("rand1", "rand2")} for c in cands}
    rep = analyze(base, arms, ctrls, keys, study["cluster_of"], GateSpec.round2(holm_m=expected["holm_m"]))
    assert json.loads(json.dumps(rep)) == expected  # exact, floats included


def test_fullcall_amendment_on_available_arms(study):
    """The full-call amendment on whatever candidate full-call rows are complete (descriptive numbers recorded in
    the study's STATUS for opus_r2: accepted 444 -> 443 of 640, compile-fail 7.9 -> 8.4 %, old_string artefacts
    89 -> 65, new_string artefacts 67 -> 47, exact copies 356 -> 381)."""
    g, keys = study["g"], study["keys"]
    registered = json.loads((g / "gate_r2_report.json").read_text())
    base = read_jsonl_unique(g / "fullcall" / "gen_base.scored.jsonl")
    cands = {}
    for c in registered["candidates"]:
        p = g / "fullcall" / f"gen_{c}.scored.jsonl"
        if p.exists():
            rows = read_jsonl_unique(p)
            if set(rows) == set(keys):
                cands[c] = rows
    if not cands:
        pytest.skip("no complete candidate full-call rows yet")
    out = amend(registered, base, cands, keys, strict=False)
    b = fullcall_summary(base, keys)
    assert round(b["accepted"] * 640) == 444 and round(b["compile_fail_rate_among_accepted"], 3) == 0.079
    assert round(b["artefact_old"] * 640) == 89 and round(b["artefact_new"] * 640) == 67 and round(b["exact_copy"] * 640) == 356
    if "opus_r2" in cands:
        o = out["candidates"]["opus_r2"]
        f = o["fullcall"]
        assert round(f["accepted"] * 640) == 443 and round(f["compile_fail_rate_among_accepted"], 3) == 0.084
        assert round(f["artefact_old"] * 640) == 65 and round(f["artefact_new"] * 640) == 47 and round(f["exact_copy"] * 640) == 381
        assert o["fullcall_compile_guard_pass"] is True
        assert o["registered"]["compile_guard_pass"] is False
        assert o["amended_primary_pass"] is False  # the served-ok guard still fails
