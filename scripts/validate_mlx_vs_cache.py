"""Real-model check 1: the library's MLX backend reproduces the study server's cached base generations.

NOT run in CI. Needs Gemma 4 E4B (about 16 GB in bf16) and a Mac with enough free unified memory; do not run it
while other model processes use the GPU.

    python scripts/validate_mlx_vs_cache.py \\
        --model-dir  .../data/models/gemma-4-e4b-it-qat-q4_0-unquantized \\
        --calls      .../outputs/weight_edit/shared/search_r2/calls.jsonl \\
        --cache      .../outputs/weight_edit/wsp_cache/base_gen.jsonl \\
        [--fullcall-cache .../outputs/weight_edit/wsp_cache/base_fullcall.jsonl] \\
        --n 40 --out /tmp/mlx_vs_cache.json

The cache holds (key, draw, gen) rows the study's forward-pass server wrote with mlx_lm's generate_step, bf16,
prefill chunks of 2048, seeds seed_of(key, draw) (draw 0 greedy, draw 1 sampled at T 0.6 / top_p 0.95 / top_k 64).
This script regenerates the same (key, draw) pairs with weightprog's MLX backend under the same settings and reports
the share of token-identical generations, greedy and sampled separately. Expected: all or nearly all identical
(the explicit decode loop equals generate_step on the tiny model; bf16 kernels are deterministic on one machine).
A mismatch rate above a few percent means the backend is a different instrument from the study's server.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--calls", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--fullcall-cache")
    ap.add_argument("--n", type=int, default=40, help="calls to check (random sample of cached keys, seed 0)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from weightprog.backends import get_backend, seed_of
    from weightprog.replay import frozen_gen, fullcall_gen, load_calls

    calls = {c["key"]: c for c in load_calls(a.calls)}
    cache = {}
    for line in open(a.cache):
        g = json.loads(line)
        cache[(g["key"], g["draw"])] = g
    keys = sorted({k for k, _ in cache if k in calls})
    random.Random(0).shuffle(keys)
    keys = keys[: a.n]
    t0 = time.time()
    be = get_backend("mlx", a.model_dir, dtype="bfloat16")
    print(f"model loaded in {time.time() - t0:.0f}s; checking {len(keys)} calls", flush=True)
    rows = []
    for i, k in enumerate(keys):
        draws = sorted(d for (kk, d) in cache if kk == k)
        n = max(draws) + 1
        got = frozen_gen(be, calls[k], n, [seed_of(k, d) for d in range(n)])
        for d in draws:
            rows.append({"key": k, "draw": d, "same": int(got[d]["gen"] == cache[(k, d)]["gen"]),
                         "ours": got[d]["gen"][:200], "cached": cache[(k, d)]["gen"][:200]})
        print(f"[{i + 1}/{len(keys)}] {k}: " + " ".join(str(r["same"]) for r in rows[-len(draws):]), flush=True)
    rep = {"n": len(rows)}
    for d in (0, 1):
        rr = [r for r in rows if r["draw"] == d]
        if rr:
            rep[f"identical_draw{d}"] = sum(r["same"] for r in rr) / len(rr)
    if a.fullcall_cache:
        fc = {}
        for line in open(a.fullcall_cache):
            g = json.loads(line)
            fc[(g["key"], g["draw"])] = g
        fk = [k for k in keys if (k, 0) in fc]
        same = 0
        for k in fk:
            g = fullcall_gen(be, calls[k], 1, [seed_of(k, 0)])[0]
            same += int(g["gen_new"] == fc[(k, 0)]["gen_new"] and g["gen"] == fc[(k, 0)]["gen"])
        rep["fullcall_greedy_identical"] = same / len(fk) if fk else None
        rep["fullcall_n"] = len(fk)
    rep["rows"] = rows
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}))


if __name__ == "__main__":
    main()
