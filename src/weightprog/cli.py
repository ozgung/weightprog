"""`weightprog` command line.

    weightprog tiny OUT                               write a tiny random Gemma 4 checkpoint (tests, quickstart)
    weightprog init-run --config server.json RUN --proposer P --model M --effort E --budget N [--fullcall]
    weightprog serve --config server.json [--once]    forward-pass budget server (file queue)
    weightprog packet --out PACKET --brief brief.md --call-set DIR --model-dir M [--dossier D] [--prior DIR]
    weightprog brief --set KEY=VALUE ... --out brief.md [--template T]
    weightprog validate-edit EDIT --model-dir M       check an edit directory (shapes, rank, finite, no rescale)
    weightprog controls SRC OUT --seed S [--support-matched]
    weightprog replay {frozen,fullcall} --backend mlx --model-dir M --calls C --keys K --out GEN [--edit E]
    weightprog score {frozen,fullcall} GEN --calls C --files F [--replacer exact|swegemma]
    weightprog gate --gate-dir G --keys K --calls C --files F [--m 2] [--spec S | --preset round2] [--report NAME] CAND ...
    weightprog amend --gate-dir G --keys K --report NAME [--out NAME] CAND ...
    weightprog export-peft EDIT OUT [--base-model ID] [--text-only] [--bf16]
    weightprog audit RUN --allow PATH ...
    weightprog prereg | weightprog spec               print the pre-registration template / the server spec
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _calls_light(path: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["key"]] = {k: r.get(k) for k in ("repo", "file_sha", "task", "instance_id")}
    return out


def cmd_tiny(a):
    from .testing import write_tiny_model
    print(write_tiny_model(a.out, seed=a.seed, dtype=a.dtype))


def cmd_init_run(a):
    from .proposer.server import ServerConfig, init_run
    print(init_run(ServerConfig.load(a.config), a.run_id, a.proposer, a.model, a.effort, a.budget, a.fullcall))


def cmd_serve(a):
    from .proposer.server import Server, ServerConfig
    Server(ServerConfig.load(a.config)).loop(once=a.once)


def cmd_packet(a):
    from .proposer.packet import build_packet
    from .proposer.server import LIMITS
    limits = json.loads(Path(a.limits).read_text()) if a.limits else LIMITS
    print(build_packet(a.out, Path(a.brief).read_text(), a.call_set, a.model_dir, a.dossier, a.prior, limits))


def cmd_brief(a):
    from .proposer.packet import brief_template, fill_brief
    vals = dict(s.split("=", 1) for s in a.set)
    tpl = Path(a.template).read_text() if a.template else brief_template()
    Path(a.out).write_text(fill_brief(tpl, vals, allow_missing=a.allow_missing))
    print(a.out)


def cmd_validate(a):
    from .edits import EditSet
    from .io.safetensors_np import SafetensorsReader
    from .names import ModelSpec
    es = EditSet.load(a.edit)
    with SafetensorsReader(a.model_dir) as r:
        print(json.dumps(es.validate(ModelSpec.from_config(a.model_dir), r), indent=1))


def cmd_controls(a):
    from .controls import write_control
    print(json.dumps(write_control(a.src, a.out, a.seed, a.support_matched), indent=1))


def cmd_replay(a):
    from .backends import get_backend
    from .edits import EditSet
    from .replay import load_calls, read_keys, run_replay
    be = get_backend(a.backend, a.model_dir, dtype=a.dtype, device=a.device)
    calls = load_calls(a.calls, read_keys(a.keys) if a.keys else None)
    if a.limit:
        calls = calls[: a.limit]
    es = EditSet.load(a.edit) if a.edit else None
    rows = run_replay(be, calls, a.mode, a.draws, a.seed_policy, es, a.edit_label or (str(a.edit) if a.edit else None), a.out,
                      progress=lambda i, n, c: print(f"[{i + 1}/{n}] {c['key']}", flush=True))
    print(f"{len(rows)} rows -> {a.out}")


def cmd_score(a):
    from .replay import FileStore, load_calls, score_rows
    from .scoring import EditScorer, ExactReplacer, SwegemmaReplacer
    calls = {c["key"]: c for c in load_calls(a.calls, only_usable=False)}
    rep = SwegemmaReplacer() if a.replacer == "swegemma" else ExactReplacer()
    gens = [json.loads(l) for l in Path(a.gen).read_text().splitlines() if l.strip()]
    rows = score_rows(gens, calls, EditScorer(rep, file_text=FileStore(a.files)), a.mode)
    out = Path(a.out) if a.out else Path(a.gen).with_suffix(".scored.jsonl")
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    acc = sum(r["accepted"] for r in rows)
    print(f"scored {len(rows)} -> {out}: accepted {acc / max(1, len(rows)):.3f}")


def _gate_inputs(a):
    from .gate import universe
    from .replay.records import FileStore
    from .scoring import escape_artefact
    keys = universe(a.keys, tuple(range(a.draws)))
    calls = _calls_light(a.calls)
    store = FileStore(a.files)
    metrics = {"escape_artefact": lambda g: escape_artefact(g["gen"], store({"file_sha": calls[g["key"]]["file_sha"]}))}
    return keys, calls, metrics


def cmd_gate(a):
    from .gate import GateSpec, analyze, load_arm, verdict_lines, write_report
    spec = GateSpec.from_json(json.loads(Path(a.spec).read_text())) if a.spec else (GateSpec.round2() if a.preset == "round2" else GateSpec())
    if a.m:
        spec.holm_m = a.m
    keys, calls, metrics = _gate_inputs(a)
    want = {k for k, _ in keys}
    cluster_of = {k.split("|")[1] if "|" in k else k: calls[k]["repo"] for k in want}
    g = Path(a.gate_dir)
    base = load_arm(g, "base", keys, False, metrics)
    cands = {c: load_arm(g, c, keys, True, metrics) for c in a.cands}
    ctrls = {c: {x: load_arm(g, f"{c}_{x}", keys, True, metrics) for x in spec.controls} for c in a.cands}
    rep = analyze(base, cands, ctrls, keys, cluster_of, spec)
    out = write_report(rep, g / a.report)
    print("\n".join(verdict_lines(rep, spec)))
    print(f"report -> {out}")


def cmd_amend(a):
    from .gate import amend, universe
    from .gate.io import read_jsonl_unique
    g = Path(a.gate_dir)
    reg = json.loads((g / a.report).read_text())
    keys = universe(a.keys, tuple(range(a.draws)))
    base = read_jsonl_unique(g / "fullcall" / "gen_base.scored.jsonl")
    cands = {c: read_jsonl_unique(g / "fullcall" / f"gen_{c}.scored.jsonl") for c in a.cands}
    out = amend(reg, base, cands, keys, registered_name=a.report, strict=not a.no_strict)
    name = a.out or a.report.replace(".json", "_amended.json")
    (g / name).write_text(json.dumps(out, indent=1))
    for c, x in out["candidates"].items():
        print(f"{c}: full-call compile guard {x['fullcall_compile_guard_pass']} -> AMENDED PRIMARY {x['amended_primary_pass']}, "
              f"SECONDARY {x['amended_secondary_pass']}")
    print(f"report -> {g / name}")


def cmd_export_peft(a):
    from .edits import EditSet, export_peft
    print(export_peft(EditSet.load(a.edit), a.out, a.base_model or "", "model." if a.text_only else "model.language_model.",
                      dtype="BF16" if a.bf16 else "F32"))


def cmd_audit(a):
    from .proposer.audit import audit
    rep = audit(a.run, a.allow)
    print(json.dumps({"run": rep["run"], "tool_calls": rep["tool_calls"], "counts": rep["counts"]}))


def cmd_prereg(a):
    from .gate import prereg_template
    sys.stdout.write(prereg_template())


def cmd_spec(a):
    from .proposer.server import spec_text
    sys.stdout.write(spec_text())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="weightprog", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("tiny"); s.add_argument("out"); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--dtype", default="F32", choices=["F32", "BF16"]); s.set_defaults(fn=cmd_tiny)

    s = sub.add_parser("init-run"); s.add_argument("--config", required=True); s.add_argument("run_id")
    s.add_argument("--proposer", required=True); s.add_argument("--model", required=True); s.add_argument("--effort", required=True)
    s.add_argument("--budget", type=int, default=10); s.add_argument("--fullcall", action="store_true"); s.set_defaults(fn=cmd_init_run)

    s = sub.add_parser("serve"); s.add_argument("--config", required=True); s.add_argument("--once", action="store_true")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("packet"); s.add_argument("--out", required=True); s.add_argument("--brief", required=True)
    s.add_argument("--call-set", required=True); s.add_argument("--model-dir", required=True); s.add_argument("--dossier")
    s.add_argument("--prior"); s.add_argument("--limits"); s.set_defaults(fn=cmd_packet)

    s = sub.add_parser("brief"); s.add_argument("--set", action="append", default=[]); s.add_argument("--out", required=True)
    s.add_argument("--template"); s.add_argument("--allow-missing", action="store_true"); s.set_defaults(fn=cmd_brief)

    s = sub.add_parser("validate-edit"); s.add_argument("edit"); s.add_argument("--model-dir", required=True)
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("controls"); s.add_argument("src"); s.add_argument("out"); s.add_argument("--seed", type=int, required=True)
    s.add_argument("--support-matched", action="store_true"); s.set_defaults(fn=cmd_controls)

    s = sub.add_parser("replay"); s.add_argument("mode", choices=["frozen", "fullcall"])
    s.add_argument("--backend", default="mlx", choices=["mlx", "torch"]); s.add_argument("--model-dir", required=True)
    s.add_argument("--dtype", default="bfloat16"); s.add_argument("--device"); s.add_argument("--calls", required=True)
    s.add_argument("--keys"); s.add_argument("--draws", type=int, default=2)
    s.add_argument("--seed-policy", default="index", choices=["index", "key"]); s.add_argument("--edit")
    s.add_argument("--edit-label"); s.add_argument("--limit", type=int); s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("score"); s.add_argument("mode", choices=["frozen", "fullcall"]); s.add_argument("gen")
    s.add_argument("--calls", required=True); s.add_argument("--files", required=True)
    s.add_argument("--replacer", default="exact", choices=["exact", "swegemma"]); s.add_argument("--out")
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("gate"); s.add_argument("--gate-dir", required=True); s.add_argument("--keys", required=True)
    s.add_argument("--calls", required=True); s.add_argument("--files", required=True); s.add_argument("--draws", type=int, default=2)
    s.add_argument("--m", type=int); s.add_argument("--spec"); s.add_argument("--preset", choices=["round2"])
    s.add_argument("--report", default="gate_report.json")
    s.add_argument("cands", nargs="+"); s.set_defaults(fn=cmd_gate)

    s = sub.add_parser("amend"); s.add_argument("--gate-dir", required=True); s.add_argument("--keys", required=True)
    s.add_argument("--report", required=True); s.add_argument("--out"); s.add_argument("--draws", type=int, default=2)
    s.add_argument("--no-strict", action="store_true"); s.add_argument("cands", nargs="+"); s.set_defaults(fn=cmd_amend)

    s = sub.add_parser("export-peft"); s.add_argument("edit"); s.add_argument("out"); s.add_argument("--base-model")
    s.add_argument("--text-only", action="store_true"); s.add_argument("--bf16", action="store_true")
    s.set_defaults(fn=cmd_export_peft)

    s = sub.add_parser("audit"); s.add_argument("run"); s.add_argument("--allow", action="append", default=[])
    s.set_defaults(fn=cmd_audit)

    s = sub.add_parser("prereg"); s.set_defaults(fn=cmd_prereg)
    s = sub.add_parser("spec"); s.set_defaults(fn=cmd_spec)
    return ap


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
