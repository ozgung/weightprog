"""Real-model check 2: the PyTorch backend agrees with the MLX backend on Gemma 4 E4B (greedy text, log-probs).

NOT run in CI. Two steps, each on a machine that can hold the model (MLX on Apple silicon; torch on a CUDA GPU,
e.g. a Kaggle or Colab L4/T4x2/A100):

    # 1. on each machine, the same calls and seeds
    python scripts/validate_torch_vs_mlx.py run --backend mlx   --model-dir M --calls C --keys K --n 40 --out mlx.jsonl
    python scripts/validate_torch_vs_mlx.py run --backend torch --model-dir M --calls C --keys K --n 40 --out torch.jsonl \\
        [--device cuda]
    # 2. anywhere
    python scripts/validate_torch_vs_mlx.py compare mlx.jsonl torch.jsonl

Each `run` writes, per call: the greedy frozen-prefix generation (draw 0), the greedy full-call pair, and the
native log-probs of the recorded old_string (teacher-forced). `compare` reports greedy text identity, outcome
identity is left to the scorer, and the max / median absolute log-prob difference. The study's own port check
(an earlier version of this backend on a Colab L4, 40 calls) found greedy text identical 39/40 and log-probs within
0.12 nats max, median 0.000. Sampled draws are not compared: the RNGs differ by design.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time


def cmd_run(a) -> None:
    from weightprog.backends import get_backend
    from weightprog.replay import fullcall_gen, frozen_gen, load_calls, read_keys

    calls = load_calls(a.calls, read_keys(a.keys) if a.keys else None)[: a.n]
    t0 = time.time()
    be = get_backend(a.backend, a.model_dir, dtype="bfloat16", device=a.device)
    print(f"{a.backend} loaded in {time.time() - t0:.0f}s", flush=True)
    tok = be.tokenizer
    with open(a.out, "w") as f:
        for i, c in enumerate(calls):
            g = frozen_gen(be, c, 1, [0])[0]
            try:
                fc = fullcall_gen(be, c, 1, [0])[0]
            except ValueError:
                fc = None
            p_ids = tok.encode(c["prompt"] + c["prefix"])
            c_ids = tok.encode(c["old_string"])[:256]
            ids = p_ids + c_ids
            pos = list(range(len(p_ids) - 1, len(ids) - 1))
            lp = be.logprobs(ids, pos)
            tgt = [float(lp[j, t]) for j, t in enumerate(c_ids)]
            f.write(json.dumps({"key": c["key"], "gen": g["gen"], "fc": fc and [fc["gen_new"], fc["gen"]], "lp": tgt,
                                "backend": be.name}) + "\n")
            print(f"[{i + 1}/{len(calls)}] {c['key']}", flush=True)


def cmd_compare(a) -> None:
    A = {r["key"]: r for r in map(json.loads, open(a.a))}
    B = {r["key"]: r for r in map(json.loads, open(a.b))}
    ks = sorted(set(A) & set(B))
    same = sum(A[k]["gen"] == B[k]["gen"] for k in ks)
    fck = [k for k in ks if A[k]["fc"] and B[k]["fc"]]
    same_fc = sum(A[k]["fc"] == B[k]["fc"] for k in fck)
    diffs = [abs(x - y) for k in ks for x, y in zip(A[k]["lp"], B[k]["lp"])]
    print(json.dumps({"calls": len(ks), "greedy_identical": same, "fullcall_identical": same_fc, "fullcall_n": len(fck),
                      "logprob_absdiff_max": max(diffs) if diffs else None,
                      "logprob_absdiff_median": statistics.median(diffs) if diffs else None}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--backend", required=True, choices=["mlx", "torch"])
    r.add_argument("--model-dir", required=True)
    r.add_argument("--calls", required=True)
    r.add_argument("--keys")
    r.add_argument("--n", type=int, default=40)
    r.add_argument("--device")
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_compare(a)


if __name__ == "__main__":
    main()
