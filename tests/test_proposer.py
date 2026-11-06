"""Proposer kit on the tiny model: kernel, budget server (queue, budget, pinning, failures), packet, brief, audit."""

import json
import subprocess
import sys

import numpy as np
import pytest

from weightprog.edits import EditSet
from weightprog.proposer.audit import audit
from weightprog.proposer.kernel import Kernel
from weightprog.proposer.packet import brief_template, build_packet, fill_brief
from weightprog.proposer.server import Server, ServerConfig, init_run, spec_text
from weightprog.testing.synthetic import synthetic_calls, write_call_set

DOWN5 = "model.language_model.layers.5.mlp.down_proj.weight"


def test_kernel(tiny_dir, tmp_path, mlx_be):
    K = Kernel(tiny_dir)
    assert DOWN5 in K.names(r"layers\.5\.mlp") and K.shape(DOWN5) == (64, 96)
    np.testing.assert_array_equal(K.load(DOWN5), mlx_be.read_weight(DOWN5))
    assert K.allowed(DOWN5) and not K.allowed("model.language_model.norm.weight")
    assert not K.allowed("model.language_model.layers.4.self_attn.k_proj.weight")
    run = tmp_path / "run"
    rng = np.random.default_rng(0)
    meta = K.save_edit(run, DOWN5, rng.standard_normal((64, 2)), rng.standard_normal((2, 96)), note="test")
    assert meta["rank"] == 2 and (run / "edit" / "layers_5_mlp_down_proj.npz").exists()
    es = EditSet.load_dir(run / "edit")
    assert es[DOWN5].meta["note"] == "test"
    with pytest.raises(ValueError, match="rank"):
        K.save_edit(run, DOWN5, np.ones((64, 129)), np.ones((129, 96)))
    with pytest.raises(ValueError, match="rescale"):
        U, S, Vt = np.linalg.svd(K.load(DOWN5), full_matrices=False)
        K.save_edit(run, DOWN5, 0.5 * U * S, Vt)
    assert [m["module"] for m in K.list_edits(run)] == [DOWN5]
    assert K.remove_edit(run, DOWN5) and K.list_edits(run) == []
    log = [json.loads(l) for l in (run / "kernel_log.jsonl").read_text().splitlines()]
    assert [r["op"] for r in log] == ["save_edit", "remove_edit"]


@pytest.fixture
def server_env(tmp_path, be, tiny_dir):
    calls, files = synthetic_calls(6, seed=5)
    cs = write_call_set(tmp_path / "callset", calls, files)
    cfg = ServerConfig(runs_dir=str(tmp_path / "runs"), calls=str(cs / "calls.jsonl"), files=str(cs / "files"),
                       cache_dir=str(tmp_path / "cache"), model_dir=str(tiny_dir), backend=be.name)
    (tmp_path / "runs").mkdir()
    return cfg, Server(cfg, backend=be), [c["key"] for c in calls]


def req(run, k, body):
    (run / "requests" / f"fwd_{k}.json").write_text(json.dumps(body))


def done(run, k):
    return json.loads((run / "requests" / f"fwd_{k}.done.json").read_text())


def test_server_queue_budget_and_pinning(server_env, tiny_dir):
    cfg, srv, keys = server_env
    run = init_run(cfg, "r1", "tester", "none", "low", budget=3, fullcall=True)
    req(run, 1, {"type": "gen", "arms": ["base"], "calls": keys[:3], "draws": 2})
    req(run, 2, {"type": "gen", "arms": ["base", "edit"], "calls": keys[:2]})  # no edit saved: fails, not counted
    req(run, 10, {"type": "read", "texts": [{"key": keys[0]}, {"key": keys[1], "continuation": "x = 1"}],
                  "layers": [4, 5], "capture": ["resid", "mlp", "attn", "logprobs"], "attn_layers": [5]})
    recs = srv.poll_once()
    assert [r["request"] for r in recs] == ["fwd_1.json", "fwd_2.json", "fwd_10.json"]  # numeric order
    d1 = done(run, 1)
    assert d1["status"] == "ok" and d1["passes_used"] == 1
    a = d1["result"]["arms"]["base"]
    assert a["n"] == 6 and 0 <= a["escape_artefact"] <= 1 and "by_recorded_stratum" in a
    assert d1["result"]["fullcall"]["arms"]["base"]["n"] == 6
    assert done(run, 2)["status"].startswith("failed") and "no saved edit" in done(run, 2)["error"]
    d10 = done(run, 10)
    assert d10["status"] == "ok" and d10["passes_used"] == 2
    z = np.load(run / "work" / "fwd_10" / "text_1.npz")
    assert z["resid"].shape[0] == 2 and z["attn"].shape[0] == 1 and z["mlp"].dtype == np.float16
    assert len(z["logprobs"]) == len(z["positions"]) and z["top5_ids"].shape[1] == 5
    # an edit, pinned by its snapshot
    K = Kernel(tiny_dir)
    rng = np.random.default_rng(1)
    K.save_edit(run, DOWN5, rng.standard_normal((64, 1)) * 2, rng.standard_normal((1, 96)))
    snap = EditSet.snapshot(run / "edit")
    req(run, 11, {"type": "gen", "arms": ["base", "edit"], "calls": keys[:3], "expect_edit": {"other.npz": "0"}})
    req(run, 12, {"type": "gen", "arms": ["base", "edit"], "calls": keys[:3], "expect_edit": snap, "draws": 2})
    req(run, 13, {"type": "gen", "arms": ["base"], "calls": keys[:1]})
    srv.poll_once()
    assert "pinned snapshot mismatch" in done(run, 11)["error"]
    d12 = done(run, 12)
    assert d12["status"] == "ok" and d12["edit_snapshot"] == snap and d12["passes_used"] == 3
    assert sum(d12["result"]["paired_accepted"].values()) == 6
    assert done(run, 13)["status"] == "refused"  # budget 3 used
    state = json.loads((run / "state.json").read_text())
    assert state["used"] == 3 and [p["request"] for p in state["passes"]] == ["fwd_1.json", "fwd_10.json", "fwd_12.json"]
    # the base cache made the second base arm free: same generations
    rows = [json.loads(l) for l in (run / "work" / "fwd_12" / "rows.jsonl").read_text().splitlines()]
    base1 = [json.loads(l) for l in (run / "work" / "fwd_1" / "rows.jsonl").read_text().splitlines()]
    assert [r["gen"] for r in rows if r["arm"] == "base"] == [r["gen"] for r in base1]


def test_server_limits(server_env):
    cfg, srv, keys = server_env
    run = init_run(cfg, "r2", "tester", "none", "low", budget=5)
    req(run, 1, {"type": "gen", "arms": ["base"], "calls": ["not-a-key"]})
    req(run, 2, {"type": "gen", "arms": ["base"], "calls": keys[:1], "draws": 3})
    req(run, 3, {"type": "read", "texts": [{"key": keys[0]}], "layers": list(range(9))})
    req(run, 4, {"type": "train"})
    srv.poll_once()
    for k in (1, 2, 3, 4):
        assert done(run, k)["status"].startswith("failed")
    assert json.loads((run / "state.json").read_text())["used"] == 0


def test_packet_and_brief(tmp_path, tiny_dir):
    calls, files = synthetic_calls(3, seed=6)
    cs = write_call_set(tmp_path / "cs", calls, files)
    vals = {k: "x" for k in ("RUN_DIR", "BUDGET", "PROPOSER", "MODEL", "EFFORT", "MODEL_NAME", "SYSTEM", "BEHAVIOUR",
                             "TARGET", "GUARDRAIL", "FAITHFULNESS", "JUDGING_SUMMARY", "PYTHON", "PACKET", "N_EDITABLE",
                             "MODEL_DIR", "ARCH_NOTES", "MAX_PY_RUNS", "PASS_TIMING", "WAIT_NOTE", "GEN_CALLS",
                             "METRIC_NAME", "READ_TEXTS", "READ_LAYERS", "ATTN_LAYERS", "DATA")}
    with pytest.raises(ValueError, match="unfilled"):
        fill_brief(brief_template(), {"RUN_DIR": "r"})
    text = fill_brief(brief_template(), vals)
    assert '"expect_edit": {file name: sha256}' in text  # literal braces survive
    pk = build_packet(tmp_path / "packet", text, cs, tiny_dir, limits={"gen_calls": 7})
    assert (pk / "search" / "calls.jsonl").exists() and "gen_calls" in (pk / "server_spec.md").read_text()
    # the packet's kernel runs stand-alone (no weightprog import), from a fresh interpreter in the packet
    code = ("import sys; sys.path.insert(0, %r); import kernel as K; "
            "print(K.shape('model.language_model.layers.5.mlp.down_proj.weight'))") % str(pk)
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=pk / "scratch")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "(64, 96)"
    with pytest.raises(PermissionError):
        (pk / "brief.md").write_text("tamper")
    (pk / "scratch" / "note.txt").write_text("ok")
    build_packet(tmp_path / "packet", text, cs, tiny_dir)  # rebuild over a read-only packet works
    assert "LIMITS" in spec_text()


def test_audit_flags(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    ev = [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "1", "name": "Bash", "input": {"command": "curl https://example.com"}},
            {"type": "tool_use", "id": "2", "name": "Read", "input": {"file_path": "/Users/someone/secret.txt"}},
            {"type": "tool_use", "id": "3", "name": "WebFetch", "input": {"url": "x"}},
            {"type": "tool_use", "id": "4", "name": "Bash", "input": {"command": "python -c 'from mlx_lm.utils import load_model'"}},
            {"type": "tool_use", "id": "5", "name": "Read", "input": {"file_path": str(run / "notebook.md")}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "2", "content": "Operation not permitted"}]}},
        {"type": "item.completed", "item": {"type": "command_execution", "command": "ls /Users/other", "aggregated_output": ""}},
        {"type": "error", "message": "a string message"},
    ]
    (run / "transcript_1.jsonl").write_text("\n".join(json.dumps(e) for e in ev) + "\nnot json\n")
    rep = audit(run, allowed=[str(tmp_path / "packet")])
    c = rep["counts"]
    assert rep["tool_calls"] == 6
    assert c["network"] == 1 and c["tools"] == 1 and c["model_load"] == 1 and c["denials"] == 1
    assert {p[2] for p in rep["paths"]} == {"/Users/someone/secret.txt", "/Users/other"}
    assert (run / "audit.json").exists()
