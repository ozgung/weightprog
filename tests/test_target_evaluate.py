"""Cases, judging, chat rendering, the evaluate loop on the tiny model, and the compare rule on synthetic rows."""

import json
import sys
import types

import numpy as np
import pytest

from weightprog.controls import matched_control
from weightprog.edits import EditSet, ModuleEdit
from weightprog.evaluate import Criteria, compare, run, seq_logprob, side_effects
from weightprog.names import module_name
from weightprog.prompts import render_chat
from weightprog.target import Case, check_splits, cut_at_stop, judge, load_cases, save_cases, select


def test_case_fields_and_judge(tmp_path):
    c = Case(id="a", prompt="x = obj.", good=[r"model_dump\("], bad=r"\.dict\(", max_new=8)
    assert c.mode == "judged" and c.bad == [r"\.dict\("] and c.group == "a"
    assert judge("y = obj.dict()", c) == "bad"
    assert judge("y = obj.model_dump()", c) == "good"
    assert judge("y = obj.model_dump() or obj.dict()", c) == "bad"  # bad wins
    assert judge("pass", c) == "other"
    assert Case(id="k", prompt="p", role="keep").mode == "unchanged"
    assert Case(id="m", prompt="p", contrast={"good": "a", "bad": "b"}).mode == "contrast"
    with pytest.raises(ValueError):
        Case(id="z")
    with pytest.raises(ValueError):
        Case(id="z", prompt="p", role="other")
    cs = [c, Case(id="b", prompt="q", split="holdout", group="a")]
    save_cases(cs, tmp_path / "c.jsonl")
    back = load_cases(tmp_path / "c.jsonl")
    assert [x.id for x in back] == ["a", "b"] and back[0].bad == c.bad
    assert check_splits(back) == ["a"]  # group "a" in both splits: leak
    assert [x.id for x in select(back, split="holdout")] == ["b"]
    (tmp_path / "dup.jsonl").write_text(json.dumps({"id": "a", "prompt": "p"}) + "\n" + json.dumps({"id": "a", "prompt": "q"}) + "\n")
    with pytest.raises(ValueError):
        load_cases(tmp_path / "dup.jsonl")


def test_check_function_and_stop():
    mod = types.ModuleType("wp_test_checks")
    mod.has_two = lambda text, case: "2" in text
    mod.three = lambda text, case: "other"
    sys.modules["wp_test_checks"] = mod
    assert judge("a2", Case(id="a", prompt="p", check="wp_test_checks:has_two")) == "good"
    assert judge("a", Case(id="a", prompt="p", check="wp_test_checks:has_two")) == "bad"
    assert judge("a", Case(id="a", prompt="p", check="wp_test_checks:three")) == "other"
    assert cut_at_stop("abc\n```\nrest", ["```", "zz"]) == "abc\n"


def test_render_chat_fallback_and_template(tmp_path):
    m = [{"role": "user", "content": "Hi"}]
    assert render_chat(m) == "<bos><|turn>user\nHi<turn|>\n<|turn>model\n"
    assert render_chat(m, prefill="```python\n").endswith("<|turn>model\n```python\n")
    (tmp_path / "chat_template.jinja").write_text(
        "{{ bos_token }}{% for m in messages %}[{{ m['role'] }}]{{ m['content'] }}{% endfor %}"
        "{% if add_generation_prompt %}[model]{% endif %}")
    pytest.importorskip("jinja2")
    assert render_chat(m, tmp_path) == "<bos>[user]Hi[model]"


def _strong_edit(be, seed=0, scale=30.0):
    L = be.spec.n_layers - 1
    name = module_name(L, "down_proj")
    W = be.read_weight(name)
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((W.shape[0], 2)).astype(np.float32) * scale / np.sqrt(W.shape[0])
    A = rng.standard_normal((2, W.shape[1])).astype(np.float32) / np.sqrt(W.shape[1])
    return EditSet([ModuleEdit(name, B, A)])


def _cases():
    return [Case(id="t1", prompt="def f(x):\n    return", good=[r"x"], bad=[r"y"], max_new=6),
            Case(id="t2", prompt="class A:\n    def g(self):", good=[r"self"], bad=[r"pass"], max_new=6),
            Case(id="k1", prompt="# a comment about code\n", role="keep", max_new=5),
            Case(id="c1", prompt="a = 1\nb = ", contrast={"good": "2", "bad": "x"}, role="keep")]


def test_run_rows_and_determinism(be):
    cases = _cases()
    arms = {"base": None, "edit": _strong_edit(be)}
    rows = run(be, cases, arms, draws=2)
    assert len(rows) == 2 * (3 * 2 + 1)
    assert {r["arm"] for r in rows} == {"base", "edit"}
    gen = [r for r in rows if r["mode"] == "judged"]
    assert all(r["label"] in ("good", "bad", "other") for r in gen)
    assert rows == run(be, cases, arms, draws=2)  # same seeds, same edit: same rows
    # the strong edit changes something on this random model
    by = {(r["arm"], r["id"], r["draw"]): r for r in rows}
    assert any(by[("base", c, d)].get("text") != by[("edit", c, d)].get("text")
               for c in ("t1", "t2", "k1") for d in (0, 1)) or by[("base", "c1", 0)]["margin"] != by[("edit", "c1", 0)]["margin"]


def test_seq_logprob_matches_manual(be):
    tok = be.tokenizer
    p, c = tok.encode("def f(x):\n    return "), tok.encode("x + 1")
    lp = be.logprobs(p + c, list(range(len(p) - 1, len(p) + len(c) - 1)))
    assert seq_logprob(be, p, c) == pytest.approx(float(sum(lp[j, t] for j, t in enumerate(c))), abs=1e-5)


def test_side_effects(be):
    e = _strong_edit(be)
    tiny = e.scaled(1e-6)
    out = side_effects(be, {"base": None, "edit": e, "tiny": tiny, "rand": matched_control(e, seed=3)},
                       texts=["def f(x):\n    return x + 1\n", "Some prose about rivers and bakers."], chunk=16)
    assert set(out) == {"edit", "tiny", "rand"}
    assert out["tiny"]["kl_mean"] < 1e-6 and out["tiny"]["top1_agree"] == 1.0
    assert out["edit"]["kl_mean"] > out["tiny"]["kl_mean"]
    assert out["edit"]["n_positions"] == out["tiny"]["n_positions"] > 10


def _rows(spec):
    """spec: {(arm, id): (role, mode, [labels] | margin | text)} -> rows."""
    rows = []
    for (arm, cid), (role, mode, val) in spec.items():
        base = {"arm": arm, "id": cid, "role": role, "split": "tune", "group": cid, "mode": mode}
        if mode == "contrast":
            rows.append({**base, "draw": 0, "margin": val})
        elif mode == "unchanged":
            rows.append({**base, "draw": 0, "text": val, "label": None})
        else:
            for d, lab in enumerate(val):
                rows.append({**base, "draw": d, "text": lab, "label": lab})
    return rows


def test_compare_rule():
    spec = {}
    for i in range(8):  # edit fixes 8 target cases; base and controls are bad
        for arm in ("base", "rand1"):
            spec[(arm, f"t{i}")] = ("target", "judged", ["bad", "bad"])
        spec[("edit", f"t{i}")] = ("target", "judged", ["good", "good"])
    for arm in ("base", "rand1", "edit"):
        spec[(arm, "k1")] = ("keep", "judged", ["good", "good"])
        spec[(arm, "k2")] = ("keep", "contrast", 2.0)
        spec[(arm, "k3")] = ("keep", "unchanged", "same text")
    side = {k: {"kl_mean": 0.001, "kl_p95": 0.002, "top1_agree": 1.0, "nll_delta": 0.0, "n_positions": 10, "n_texts": 1}
            for k in ("edit", "rand1")}
    r0 = compare(_rows(spec), arm="edit", controls=["rand1"])
    assert r0["target"]["pass"] and r0["keep"]["pass"] and r0["verdict"] == "INCOMPLETE" and not r0["pass"]
    assert r0["missing"] == ["side effects"]
    r = compare(_rows(spec), side, arm="edit", controls=["rand1"])
    assert r["verdict"] == "PASS" and r["pass"]
    t = r["target"]["by_mode"]["judged"]["tests"]["base"]
    assert t["cases_better"] == 8 and t["p_one_sided"] == pytest.approx(1 / 256) and t["mean_good_change"] == 1.0
    assert compare(_rows(spec), side, arm="edit")["verdict"] == "INCOMPLETE"  # no controls
    # bad -> other is not a fix: the good rate does not rise
    other = dict(spec)
    for i in range(8):
        other[("edit", f"t{i}")] = ("target", "judged", ["other", "other"])
    r_o = compare(_rows(other), side, arm="edit", controls=["rand1"])
    assert not r_o["target"]["pass"] and r_o["target"]["by_mode"]["judged"]["tests"]["base"]["bad_to_other_cases"] == 8
    assert compare(_rows(other), side, arm="edit", controls=["rand1"],
                   criteria=Criteria(min_good_share=0.0))["target"]["pass"]
    # break one keep case in draw 1 and the contrast keep case; change the unchanged one
    spec[("edit", "k1")] = ("keep", "judged", ["good", "bad"])
    spec[("edit", "k2")] = ("keep", "contrast", -0.5)
    spec[("edit", "k3")] = ("keep", "unchanged", "other text")
    r2 = compare(_rows(spec), side, arm="edit", controls=["rand1"], criteria=Criteria(keep_max_broken=0))
    assert not r2["keep"]["pass"] and r2["verdict"] == "FAIL"
    assert r2["keep"]["broken_per_draw"] == {"0": ["k2"], "1": ["k1"]} and r2["keep"]["changed"] == ["k3"]
    assert r2["keep"]["controls"]["rand1"]["broken_per_draw"] == {"0": 0, "1": 0}
    r3 = compare(_rows(spec), side, arm="edit", controls=["rand1"], criteria=Criteria(keep_max_broken=1, keep_max_changed_frac=1.0))
    assert r3["keep"]["pass"]
    # a control that moves the metric as much as the edit: target fails against it
    moved = dict(spec)
    for i in range(8):
        moved[("rand1", f"t{i}")] = ("target", "judged", ["good", "good"])
    assert not compare(_rows(moved), side, arm="edit", controls=["rand1"])["target"]["pass"]
    # side effects over the limit fail the verdict
    big = {"edit": dict(side["edit"], kl_mean=0.2), "rand1": side["rand1"]}
    r5 = compare(_rows(spec), big, arm="edit", controls=["rand1"], criteria=Criteria(keep_max_broken=1, keep_max_changed_frac=1.0))
    assert r5["target"]["pass"] and not r5["side_effects"]["pass"] and r5["verdict"] == "FAIL"
    assert any(line.startswith("VERDICT") for line in r5["summary"])
    # one improved group of eight: the group-bootstrap CI reaches 0, so the target fails even at a loose alpha
    mixed = dict(spec)
    for i in range(7):
        mixed[("edit", f"t{i}")] = ("target", "judged", ["bad", "bad"])
    rm = compare(_rows(mixed), side, arm="edit", controls=["rand1"], criteria=Criteria(alpha=0.9, keep_max_broken=1, keep_max_changed_frac=1.0))
    assert rm["target"]["by_mode"]["judged"]["tests"]["base"]["ci95_group_bootstrap"][0] == 0.0 and not rm["target"]["pass"]


def test_reference_texts_load():
    from weightprog.evaluate import reference_texts
    f, e = reference_texts("fit"), reference_texts("eval")
    assert len(f) >= 10 and len(e) >= 20 and not set(f) & set(e)
    assert any("def " in x for x in f + e) and any("<|turn>" in x for x in e) and not any("pydantic" in x for x in f + e)


def test_probe_helpers(be):
    from weightprog.construct import TailState, predict_logit_shift
    from weightprog.probe import auc, collect_tail, divergence, margins, measure_logit_shift, readout, show_tokens
    tok = be.tokenizer
    assert "".join(p for _, p in show_tokens(tok, "abc")) == "abc"
    d = divergence(tok, "x = obj.", "model()", "dict()")
    assert d.prefix_ids == tok.encode("x = obj.") and d.good_id == tok.encode("m")[0] and d.bad_id == tok.encode("d")[0]
    with pytest.raises(ValueError):
        divergence(tok, "x", "ab", "abc")
    m = margins(be, [(d.prefix_ids, d.good_id, d.bad_id)])
    lp = be.logprobs(d.prefix_ids, [len(d.prefix_ids) - 1])[0]
    assert m[0] == pytest.approx(float(lp[d.good_id] - lp[d.bad_id]), abs=1e-5)
    L = be.spec.n_layers - 1
    a, b = tok.encode("def f(x):\n    return x"), tok.encode("class Q:\n    pass")
    st = collect_tail(be, [(a, [3, 5]), (b, [4])], L)
    assert st.act.shape[0] == 3 and st.extra["n_per_item"] == [2, 1]
    one = TailState.from_backend(be, b, [4], L)
    np.testing.assert_allclose(st.act[2], one.act[0], atol=1e-6)
    e = _strong_edit(be, scale=3.0)
    meas = measure_logit_shift(be, e, a, [3, 5], [d.good_id, d.bad_id])
    pred = predict_logit_shift(TailState.from_backend(be, a, [3, 5], L), list(e)[0], [d.good_id, d.bad_id])
    # log-prob shift = logit shift minus the log-normaliser shift: compare differences between the two tokens
    np.testing.assert_allclose(meas[:, 0] - meas[:, 1], pred[:, 0] - pred[:, 1], atol=2e-3)
    assert auc(np.array([3.0, 2.0, 1.0, 0.0]), np.array([1, 1, 0, 0])) == 1.0
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 6))
    y = (X[:, 0] > 0).astype(int)
    assert readout(X, y, lam=1e-2) > 0.9


def test_stop_string_ends_generation(be):
    from weightprog.evaluate import stop_string_ids
    tok = be.tokenizer
    assert tok.token_to_id("e") in stop_string_ids(tok, "e")
    free = run(be, [Case(id="s", prompt="def f(x):\n    return", good=["x"], max_new=24)], draws=1)[0]
    k, ch = next((k, c) for k, c in enumerate(free["text"]) if c.strip())
    stopped = run(be, [Case(id="s", prompt="def f(x):\n    return", good=["x"], max_new=24, stop=[ch])], draws=1)[0]
    assert stopped["n_tokens"] <= k + 1 and ch not in stopped["text"]
