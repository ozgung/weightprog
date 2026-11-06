"""The gate on synthetic arms (no model): pass, guard failures, Holm, BLOCK conditions, the full-call amendment."""

import json
import math
import random

import pytest

from weightprog.gate import GateBlocked, GateSpec, amend, analyze, load_arm, universe
from weightprog.stats import binom_two_sided, bootstrap_clustered_ci, bootstrap_mean_ci, holm, sign_test


def test_sign_test_and_holm():
    p, pos, neg = sign_test([1, 1, 1, 0, -1])
    assert (pos, neg) == (3, 1) and p == pytest.approx(5 / 16)
    assert sign_test([0, 0])[0] == 1.0
    assert sign_test([1] * 20)[0] == 2 ** -20
    assert binom_two_sided(0, 10) == pytest.approx(2 / 1024)
    h = holm({"a": 0.01, "b": 0.04}, m=2)
    assert h == {"a": (0.025, True), "b": (0.05, True)}
    h = holm({"a": 0.03, "b": 0.001}, m=3)
    assert h["b"] == (0.05 / 3, True) and h["a"] == (0.025, False)
    v = {f"t{i}": (i % 3) - 1.0 for i in range(30)}
    assert bootstrap_mean_ci(v, 2000, 1) == bootstrap_mean_ci(v, 2000, 1)
    lo, hi = bootstrap_clustered_ci(v, lambda t: int(t[1:]) % 5, 2000, 2)
    assert lo <= sum(v.values()) / 30 <= hi


def synth(n_tasks=40, calls=4, seed=0):
    """Base arm: recorded kinds, acceptance and escape artefacts at plausible rates."""
    rng = random.Random(seed)
    keys, base = [], {}
    for t in range(n_tasks):
        for c in range(calls):
            k = f"{'o43' if t % 2 else 'oodb'}|task{t}|{c}|0"
            kind = rng.choice(["ok", "ok", "ok", "not_found"])
            art = int(rng.random() < 0.3)
            for d in (0, 1):
                keys.append((k, d))
                base[(k, d)] = {"recorded_kind": kind, "accepted": int(kind == "ok" or rng.random() < 0.2),
                                "escape_artefact": art, "compiles": 1}
    cluster = {f"task{t}": f"repo{t // 3}" for t in range(n_tasks)}
    return sorted(keys), base, cluster


def edited(base, fix=0.8, seed=1, break_ok=0):
    rng = random.Random(seed)
    e = {k: dict(v) for k, v in base.items()}
    for k, v in e.items():
        if v["escape_artefact"] and rng.random() < fix:
            v["escape_artefact"] = 0
    broken = 0
    for k in sorted(e):
        if broken >= break_ok:
            break
        if e[k]["recorded_kind"] == "ok" and e[k]["accepted"] and k[1] == 0:
            e[k]["accepted"] = 0
            broken += 1
    return e


def test_effective_edit_passes_primary_and_controls_do_not():
    keys, base, cl = synth()
    e = edited(base)
    ctrl = {"rand1": {k: dict(v) for k, v in base.items()}, "rand2": {k: dict(v) for k, v in base.items()}}
    rep = analyze(base, {"cand": e}, {"cand": ctrl}, keys, cl, GateSpec(holm_m=2))
    r = rep["candidates"]["cand"]
    assert r["primary_escape_artefact"]["p"] < 0.025 and r["primary_holm_pass"]
    assert r["acceptance_noninferior"] and r["served_ok_rule_pass"] and r["compile_guard_pass"]
    assert r["primary_pass"] is True
    assert r["secondary_acceptance"]["p"] == r["secondary_acceptance"]["p_raw"]  # tested because primary passed
    assert r["secondary_pass"] is False  # acceptance unchanged
    assert rep["base_escape_artefacts"] == sum(base[k]["escape_artefact"] for k in keys)
    json.dumps(rep)  # serialisable


def test_guards_fail_on_broken_copies_and_compile():
    keys, base, cl = synth()
    e = edited(base, break_ok=5)
    ctrl = {"rand1": base, "rand2": base}
    r = analyze(base, {"cand": e}, {"cand": ctrl}, keys, cl)["candidates"]["cand"]
    assert r["served_ok_loss"]["draw0"]["n_lost"] == 5 and not r["served_ok_rule_pass"]
    assert r["primary_pass"] is False and r["secondary_acceptance"]["p"] == 1.0
    e2 = edited(base)
    for k in keys[:40]:
        if e2[k]["accepted"]:
            e2[k]["compiles"] = 0
    r2 = analyze(base, {"cand": e2}, {"cand": ctrl}, keys, cl)["candidates"]["cand"]
    assert r2["compile_guard_pass"] is False and r2["primary_pass"] is False


def test_null_edit_fails_and_holm_family():
    keys, base, cl = synth()
    same = {k: dict(v) for k, v in base.items()}
    ctrl = {"rand1": base, "rand2": base}
    rep = analyze(base, {"null": same, "good": edited(base)}, {"null": ctrl, "good": ctrl}, keys, cl, GateSpec(holm_m=2))
    assert rep["candidates"]["null"]["primary_escape_artefact"]["p"] == 1.0
    assert rep["candidates"]["good"]["primary_holm_threshold"] == 0.025  # the smaller p takes the strict slot
    assert rep["candidates"]["null"]["primary_pass"] is False


def test_controls_that_also_help_block_the_claim():
    """If a random edit of the same size removes artefacts too, the candidate must beat it, not just base."""
    keys, base, cl = synth()
    e = edited(base, fix=0.5, seed=3)
    ctrl = {"rand1": edited(base, fix=0.5, seed=4), "rand2": base}
    r = analyze(base, {"cand": e}, {"cand": ctrl}, keys, cl)["candidates"]["cand"]
    assert r["primary_escape_artefact"]["vs_base"]["pass"] and not r["primary_escape_artefact"]["vs_rand1"]["pass"]
    assert r["primary_pass"] is False


def write_arm(d, arm, rows, gens, edit):
    with (d / f"gen_{arm}.jsonl").open("w") as f:
        for (k, dr) in gens:
            f.write(json.dumps({"key": k, "draw": dr, "gen": "x", "edit": edit}) + "\n")
    with (d / f"gen_{arm}.scored.jsonl").open("w") as f:
        for (k, dr), v in rows.items():
            f.write(json.dumps({"key": k, "draw": dr, **v}) + "\n")


def test_block_conditions(tmp_path):
    keys, base, cl = synth(n_tasks=4)
    (tmp_path / "keys.txt").write_text("\n".join(sorted({k for k, _ in keys})))
    assert universe(tmp_path / "keys.txt") == keys
    write_arm(tmp_path, "base", base, keys, None)
    assert len(load_arm(tmp_path, "base", keys, False)) == len(keys)
    write_arm(tmp_path, "cand", base, keys[:-1], "x/cand/edit")  # truncated generation file
    with pytest.raises(GateBlocked, match="missing 1"):
        load_arm(tmp_path, "cand", keys, True)
    write_arm(tmp_path, "cand", base, keys, "x/other/edit")  # wrong edit dir
    with pytest.raises(GateBlocked, match="not generated with cand/edit"):
        load_arm(tmp_path, "cand", keys, True)
    write_arm(tmp_path, "base", base, keys, "x/base/edit")  # base with an edit
    with pytest.raises(GateBlocked, match="has an edit"):
        load_arm(tmp_path, "base", keys, False)
    with (tmp_path / "gen_dup.jsonl").open("w") as f:
        f.write(json.dumps({"key": "a", "draw": 0}) + "\n" + json.dumps({"key": "a", "draw": 0}) + "\n")
    with pytest.raises(GateBlocked, match="duplicate"):
        load_arm(tmp_path, "dup", keys, True)


def test_fullcall_amendment_flips_only_a_confounded_compile_failure():
    keys, base, cl = synth()
    e = edited(base)
    for k in keys[:60]:  # frozen-prefix replay: compile failures that the full call does not have
        if e[k]["accepted"]:
            e[k]["compiles"] = 0
    ctrl = {"rand1": base, "rand2": base}
    reg = analyze(base, {"cand": e}, {"cand": ctrl}, keys, cl)
    assert reg["candidates"]["cand"]["compile_guard_pass"] is False
    fc_base = {k: {"accepted": v["accepted"], "compiles": 1} for k, v in base.items()}
    fc_good = {k: {"accepted": v["accepted"], "compiles": 1} for k, v in e.items()}
    out = amend(reg, fc_base, {"cand": fc_good}, keys)
    assert out["candidates"]["cand"]["fullcall_compile_guard_pass"] and out["candidates"]["cand"]["amended_primary_pass"]
    fc_bad = {k: {"accepted": v["accepted"], "compiles": 0 if i < 80 else 1} for i, (k, v) in enumerate(sorted(e.items()))}
    out = amend(reg, fc_base, {"cand": fc_bad}, keys)
    assert not out["candidates"]["cand"]["amended_primary_pass"]  # a real compile regression stays a failure
    with pytest.raises(GateBlocked):
        amend(reg, fc_base, {"other": fc_good}, keys)
    assert math.isclose(out["holm_m"], 2)
