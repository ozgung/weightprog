"""`weightprog` command line.

    weightprog census --model-dir M --cases C.jsonl [--split tune] [--draws 2] [--out rows.jsonl]
                                   how the unedited model behaves on your cases (per-case labels and rates)
    weightprog eval --model-dir M --cases C.jsonl --edit EDIT [--controls 2] [--split holdout] [--side-effects]
                    [--criteria crit.json] [--report report.json] [--rows rows.jsonl] [--base-rows base.jsonl]
                                   base vs edit vs matched random controls: target, keep, side effects, verdict
                                   (--controls 0 and no --side-effects: a quick look; the verdict is then INCOMPLETE)
    weightprog side-effects --model-dir M --edit EDIT [--texts texts.json] [--controls 2]
    weightprog render --model-dir M --cases C.jsonl [--id ID]   print the exact prompt text of cases
    weightprog validate-edit EDIT --model-dir M       check an edit directory (shapes, rank, finite, no rescale)
    weightprog controls SRC OUT --seed S              write a matched random control of an edit
    weightprog export-peft EDIT OUT [--base-model ID] [--text-only] [--bf16]
    weightprog tiny OUT                               write a tiny random Gemma 4 checkpoint (tests, dry runs)
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


def _model(a):
    from .evaluate import open_model
    return open_model(a.model_dir, backend=a.backend, dtype=a.dtype, device=a.device)


def _cases(a):
    from .target import check_splits, load_cases, select
    cases = load_cases(a.cases)
    leak = check_splits(cases)
    if leak:
        print(f"WARNING: groups in both tune and holdout: {leak[:10]}", file=sys.stderr)
    cases = select(cases, split=a.split) if a.split else cases
    if a.limit:
        cases = cases[: a.limit]
    if not cases:
        raise SystemExit("no cases selected")
    return cases


def _progress(a):
    return (lambda m: print(m, file=sys.stderr, flush=True)) if a.verbose else None


def _arms(edit_dir, n_controls, seed):
    from .controls import matched_control
    from .edits import EditSet
    e = EditSet.load(edit_dir)
    arms = {"base": None, "edit": e}
    for k in range(1, n_controls + 1):
        arms[f"rand{k}"] = matched_control(e, seed=seed + k)
    return arms


def cmd_census(a):
    from .evaluate import case_scores, run, write_rows
    be, cases = _model(a), _cases(a)
    rows = run(be, cases, {"base": None}, draws=a.draws, model_dir=a.model_dir, progress=_progress(a))
    if a.out:
        write_rows(rows, a.out)
    s = case_scores(rows, "base")
    tot = collections.Counter()
    for cid, v in s.items():
        if v["mode"] == "judged":
            print(f"{cid:28} {v['role']:6} {''.join(l[0] for l in v['labels'])}  {v['greedy'][:90]!r}")
            for lab in v["labels"]:
                tot[(v["role"], lab)] += 1
        elif v["mode"] == "contrast":
            print(f"{cid:28} {v['role']:6} margin {v['margin']:+.3f}")
        else:
            print(f"{cid:28} {v['role']:6} {v['greedy'][:90]!r}")
    for role in ("target", "keep"):
        n = sum(tot[(role, l)] for l in ("good", "bad", "other"))
        if n:
            print(f"{role}: " + "  ".join(f"{l} {tot[(role, l)] / n:.3f}" for l in ("bad", "good", "other")) + f"  ({n} generations)")


def cmd_eval(a):
    from .evaluate import Criteria, compare, read_rows, reuse_rows, run, side_effects, write_rows
    be, cases = _model(a), _cases(a)
    arms = _arms(a.edit, a.controls, a.control_seed)
    base_rows = []
    if a.base_rows:  # reuse the unedited model's rows across iterations (only rows whose case did not change)
        p = Path(a.base_rows)
        cached = read_rows(p) if p.exists() else []
        base_rows, todo = reuse_rows(cached, cases, "base", a.draws, model_dir=a.model_dir)
        if todo:
            base_rows += run(be, todo, {"base": None}, draws=a.draws, model_dir=a.model_dir, progress=_progress(a))
            others = [r for r in cached if r["arm"] == "base" and r["id"] not in {c.id for c in cases}]
            write_rows(others + base_rows, p)
        print(f"base rows: {len(cases) - len(todo)} cases reused, {len(todo)} generated -> {p}", file=sys.stderr)
        arms.pop("base")
    rows = base_rows + run(be, cases, arms, draws=a.draws, model_dir=a.model_dir, progress=_progress(a))
    if a.rows:
        write_rows(rows, a.rows)
    side = None
    if a.side_effects:
        texts = json.loads(Path(a.texts).read_text())["texts"] if a.texts else None
        side = side_effects(be, arms, texts)
    crit = Criteria(**json.loads(Path(a.criteria).read_text())) if a.criteria else Criteria()
    rep = compare(rows, side, arm="edit", controls=[k for k in arms if k.startswith("rand")], criteria=crit)
    rep["edit_dir"] = str(a.edit)
    rep["cases_file"] = str(a.cases)
    rep["split"] = a.split
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1))
    print("\n".join(rep["summary"]))


def cmd_side_effects(a):
    from .evaluate import side_effects
    be = _model(a)
    texts = json.loads(Path(a.texts).read_text())["texts"] if a.texts else None
    out = side_effects(be, _arms(a.edit, a.controls, a.control_seed), texts)
    print(json.dumps(out, indent=1))


def cmd_render(a):
    from .target import load_cases
    for c in load_cases(a.cases):
        if a.id and c.id != a.id:
            continue
        print(f"===== {c.id}\n{c.text(a.model_dir)}")


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


def cmd_export_peft(a):
    from .edits import EditSet, export_peft
    print(export_peft(EditSet.load(a.edit), a.out, a.base_model or "", "model." if a.text_only else "model.language_model.",
                      dtype="BF16" if a.bf16 else "F32"))


def cmd_tiny(a):
    from .testing import write_tiny_model
    print(write_tiny_model(a.out, seed=a.seed, dtype=a.dtype))


def _model_args(s, cases=True):
    s.add_argument("--model-dir", required=True)
    s.add_argument("--backend", default="auto", choices=["auto", "mlx", "torch"])
    s.add_argument("--dtype", default="bfloat16")
    s.add_argument("--device")
    s.add_argument("-v", "--verbose", action="store_true")
    if cases:
        s.add_argument("--cases", required=True)
        s.add_argument("--split", help="only cases with this split label (e.g. tune or holdout)")
        s.add_argument("--limit", type=int)
        s.add_argument("--draws", type=int, default=2)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="weightprog", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("census", help="run the unedited model on your cases"); _model_args(s)
    s.add_argument("--out"); s.set_defaults(fn=cmd_census)

    s = sub.add_parser("eval", help="judge an edit against base and matched random controls"); _model_args(s)
    s.add_argument("--edit", required=True); s.add_argument("--controls", type=int, default=2)
    s.add_argument("--control-seed", type=int, default=20261110); s.add_argument("--side-effects", action="store_true")
    s.add_argument("--texts"); s.add_argument("--criteria"); s.add_argument("--report"); s.add_argument("--rows")
    s.add_argument("--base-rows", help="cache of the unedited model's rows: reused when the case is unchanged, else regenerated and saved")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("side-effects", help="KL / NLL of an edit on reference text"); _model_args(s, cases=False)
    s.add_argument("--edit", required=True); s.add_argument("--controls", type=int, default=0)
    s.add_argument("--control-seed", type=int, default=20261110); s.add_argument("--texts")
    s.set_defaults(fn=cmd_side_effects)

    s = sub.add_parser("render", help="print the exact prompt text of cases"); s.add_argument("--cases", required=True)
    s.add_argument("--model-dir"); s.add_argument("--id"); s.set_defaults(fn=cmd_render)

    s = sub.add_parser("validate-edit"); s.add_argument("edit"); s.add_argument("--model-dir", required=True)
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("controls"); s.add_argument("src"); s.add_argument("out"); s.add_argument("--seed", type=int, required=True)
    s.add_argument("--support-matched", action="store_true"); s.set_defaults(fn=cmd_controls)

    s = sub.add_parser("export-peft"); s.add_argument("edit"); s.add_argument("out"); s.add_argument("--base-model")
    s.add_argument("--text-only", action="store_true"); s.add_argument("--bf16", action="store_true")
    s.set_defaults(fn=cmd_export_peft)

    s = sub.add_parser("tiny"); s.add_argument("out"); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--dtype", default="F32", choices=["F32", "BF16"]); s.set_defaults(fn=cmd_tiny)
    return ap


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
