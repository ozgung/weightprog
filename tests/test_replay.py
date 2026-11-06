"""Codec, records, scoring and both replays on the tiny model with synthetic calls."""

import json

import pytest

from weightprog.replay import (FileStore, Q, call_prefix, decode_call, encode_call, load_calls, run_replay, score_rows,
                               separator, split_prefix)
from weightprog.scoring import EditScorer, ExactReplacer, compiles, escape_artefact, exact_copy
from weightprog.testing.synthetic import synthetic_calls, write_call_set


def test_codec_roundtrip():
    args = {"old_string": "a\nb", "filepath": "x.py", "new_string": "c", "count": 3, "flag": True,
            "opts": {"z": "1", "a": [1, "two"]}}
    s = encode_call("edit_file", args)
    assert s.startswith('<|tool_call>call:edit_file{count:3,filepath:<|"|>x.py<|"|>,flag:true,new_string:')
    name, back = decode_call("thinking... " + s + " trailing")
    assert name == "edit_file" and back == args
    p = call_prefix("edit_file", {"filepath": "x.py", "new_string": "N", "old_string": "O"}, "old_string")
    assert p.endswith('new_string:<|"|>N<|"|>,old_string:<|"|>')
    assert separator("old_string") == '<|"|>,old_string:<|"|>'
    with pytest.raises(ValueError):
        encode_call("f", {"s": "bad " + Q})


def test_split_prefix():
    rec = {"key": "k", "new_string": "N", "prefix": "x" + call_prefix("edit_file", {"filepath": "f", "new_string": "N", "old_string": "O"}, "old_string")}
    assert split_prefix(rec).endswith('new_string:<|"|>')
    with pytest.raises(ValueError):
        split_prefix(dict(rec, new_string="other"))


def test_metrics():
    file = 'x = "a\\n"\ny = 1\n'  # the file really contains backslash-n inside a string literal
    assert escape_artefact('y = 1\\nz', "y = 1\nz") == 1
    assert escape_artefact('x = "a\\n"', file) == 0
    assert escape_artefact('say \\"hi\\"', 'say "hi"') == 1
    assert exact_copy("y = 1", file) == 1 and exact_copy("", file) == 0
    assert compiles("def f(:\n", "a.py") == 0 and compiles("x = 1\n", "a.py") == 1 and compiles("x", "a.md") is None
    r = ExactReplacer()
    assert r("aXa", "X", "Y", "f", False).new_content == "aYa"
    assert r("aXaX", "X", "Y", "f", False).outcome == "multiple"
    assert r("aXaX", "X", "Y", "f", True).new_content == "aYaY"
    assert r("abc", "Z", "Y", "f", False).outcome == "not_found" and r("abc", "", "Y", "f", False).outcome == "empty"


def test_scorer_rows(tmp_path):
    calls, files = synthetic_calls(6, seed=2, artefact_rate=0.5)
    d = write_call_set(tmp_path, calls, files)
    cs = load_calls(d / "calls.jsonl")
    sc = EditScorer(file_text=FileStore(d / "files"))
    for c in cs:
        row = sc.score(c, c["old_string"])
        assert row["accepted"] == int(c["error_kind"] == "ok")
        text = files[c["file_sha"]]
        want = int("\\n" in c["old_string"] and "\\n" not in text)  # a file with a real backslash-n does not count
        assert row["same_as_recorded"] == 1 and row["escape_artefact"] == want
        fc = sc.score_fullcall(c, c["old_string"], c["new_string"])
        assert fc["artefact_old"] == row["escape_artefact"] and fc["same_new_as_recorded"] == 1


def test_frozen_and_fullcall_replay(be, tmp_path):
    calls, files = synthetic_calls(4, seed=3)
    d = write_call_set(tmp_path, calls, files)
    cs = load_calls(d / "calls.jsonl")
    out = tmp_path / "gen_base.jsonl"
    rows = run_replay(be, cs[:2], "frozen", draws=2, seed_policy="key", out=out)
    assert len(rows) == 4 and {r["draw"] for r in rows} == {0, 1}
    rows2 = run_replay(be, cs, "frozen", draws=2, seed_policy="key", out=out)  # resumes: only the new calls run
    assert len(rows2) == 8 and len(out.read_text().splitlines()) == 8
    assert [r["gen"] for r in rows2[:4]] == [r["gen"] for r in rows]
    fc = run_replay(be, cs, "fullcall", draws=2, seed_policy="index", edit_label="x/arm/edit")
    assert all({"gen_new", "gen", "closed_new", "closed"} <= set(r) for r in fc)
    assert all(r["edit"] == "x/arm/edit" and r["backend"] == be.name for r in fc)
    for r in fc:
        if not r["closed_new"]:
            assert r["gen"] == "" and r["closed"] == 0
    sc = EditScorer(file_text=FileStore(d / "files"))
    scored = score_rows(fc, {c["key"]: c for c in cs}, sc, mode="fullcall")
    assert len(scored) == 8 and all("compiles_frozen_new" in s for s in scored)
    json.dumps(scored)


def test_greedy_fullcall_matches_frozen_when_new_string_matches(be, tmp_path):
    """Build the recorded new_string from the model's own greedy full-call output: then the frozen-prefix greedy
    old_string must equal the full-call greedy old_string token for token (the study's smoke test)."""
    from weightprog.replay import fullcall_gen, frozen_gen

    calls, _ = synthetic_calls(3, seed=4)
    for c in calls:
        g = fullcall_gen(be, c, 1, [0])[0]
        if not g["closed_new"]:
            continue
        c2 = dict(c, new_string=g["gen_new"])
        head = split_prefix(c)
        c2["prefix"] = head + g["gen_new"] + separator("old_string")
        f = frozen_gen(be, c2, 1, [0])[0]
        assert f["gen"] == g["gen"]
