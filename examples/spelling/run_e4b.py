"""Run the context-token-choice recipe on a real Gemma 4 checkpoint and judge both of its edits on the example cases.

    python examples/spelling/run_e4b.py MODEL_DIR OUT_DIR

Executes the recipe's code block as written (docs/recipes/context-token-choice.md) in OUT_DIR, then evaluates the
gated edit and the covariance (kv) variant on examples/spelling/cases.jsonl, tune and holdout separately, each against
the base model and two matched random edits, with side effects on the built-in eval reference texts. Writes
OUT_DIR/report_<edit>_<split>.json and OUT_DIR/summary.md. Edits are saved in OUT_DIR (they are model derivatives;
do not commit them).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from weightprog.controls import matched_control
from weightprog.edits import EditSet
from weightprog.evaluate import compare, run, side_effects
from weightprog.target import load_cases, select

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    model_dir, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    recipe = (ROOT / "docs/recipes/context-token-choice.md").read_text()
    code = re.search(r"## Code\n\n```python\n(.*?)```", recipe, re.S).group(1)
    code = code.replace('"/path/to/gemma-4-e4b-it"', repr(str(model_dir)))
    os.chdir(out)
    t0 = time.time()
    ns: dict = {}
    exec(compile(code, "context-token-choice.md", "exec"), ns)  # noqa: S102
    t_recipe = time.time() - t0
    be = ns["be"]
    cases = load_cases(HERE / "cases.jsonl")
    lines = [f"# Worked example on {model_dir.name}", "", f"recipe code: {t_recipe:.0f} s", ""]
    for name, es in (("gated", EditSet([ns["edit"]])), ("kv", EditSet([ns["kv"]]))):
        es.save_dir(out / f"edit_{name}", clear=True)
        arms = {"edit": es, "rand1": matched_control(es, seed=101), "rand2": matched_control(es, seed=102)}
        side = side_effects(be, {"base": None, **arms})
        for split in ("tune", "holdout"):
            cs = select(cases, split=split)
            t1 = time.time()
            rows = run(be, cs, {"base": None, **arms}, draws=2, model_dir=model_dir)
            rep = compare(rows, side, arm="edit", controls=["rand1", "rand2"])
            (out / f"report_{name}_{split}.json").write_text(json.dumps(rep, indent=1))
            (out / f"rows_{name}_{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            lines += [f"## {name} edit, {split} ({time.time() - t1:.0f} s)", "", "```", *rep["summary"], "```", ""]
            print("\n".join(rep["summary"]), flush=True)
    (out / "summary.md").write_text("\n".join(lines))
    print(f"-> {out / 'summary.md'}")


if __name__ == "__main__":
    main()
