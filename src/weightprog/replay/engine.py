"""Replays over any backend: frozen-prefix (regenerate the last argument) and full-call (regenerate several in order).

Frozen-prefix: prompt + prefix (which ends with `old_string:<|"|>`), generate until `<|"|>` with a cap of
min(cap, 2 * len(recorded old_string tokens) + 64). Everything the model wrote before old_string, including its
new_string, stays as served. That is cheap and isolates the copy, but it pairs an edited old_string with an
unedited new_string (see weightprog.gate.fullcall for the trap that creates).

Full-call: cut the prefix after `new_string:<|"|>`, generate new_string until `<|"|>` (cap min(2048, 2n+64)),
teacher-force `<|"|>,old_string:<|"|>`, generate old_string (cap min(1024, 2n+64)); an unclosed new_string ends the
draw (old_string empty). Both arguments come from the same arm. `regenerate_args` is the general form.

Seeds: "key" = seed_of(key, draw) (the forward-pass server's), "index" = 7919 * index + draw (the study's
stand-alone gate generators). Draw 0 is greedy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..backends.base import SERVED, Force, Generate, Sampling, index_seed, seed_of
from .codec import Q, separator, split_prefix


def seeds_for(policy: str, key: str, index: int, draws: int) -> list[int]:
    if policy == "key":
        return [seed_of(key, d) for d in range(draws)]
    if policy == "index":
        return [index_seed(index, d) for d in range(draws)]
    raise ValueError("seed policy must be key or index")


def frozen_gen(be, call: dict, draws: int, seeds: Sequence[int], max_new_cap: int = 1024, prompt_key: str = "prompt",
               sampling: Sampling = SERVED) -> list[dict]:
    tok = be.tokenizer
    q = tok.token_to_id(Q)
    ids = tok.encode(call[prompt_key] + call["prefix"])
    max_new = min(max_new_cap, 2 * len(tok.encode(call["old_string"])) + 64)
    outs = be.generate(ids, max_new, draws=draws, seeds=seeds, stop_ids=[q], sampling=sampling)
    return [{"key": call["key"], "draw": d, "gen": tok.decode(o.tokens), "closed": o.closed, "n_gen": len(o.tokens),
             "prompt_tokens": len(ids)} for d, o in enumerate(outs)]


def regenerate_args(be, text: str, arg_names: Sequence[str], caps: Sequence[int], draws: int, seeds: Sequence[int],
                    sampling: Sampling = SERVED) -> list[list[tuple[str, int, int]]]:
    """text ends with `arg_names[0]:<|"|>`; generate each argument in order, teacher-forcing the separator between
    them. Returns per draw [(text, closed, n_tokens), ...] (arguments after an unclosed one are ("", 0, 0))."""
    tok = be.tokenizer
    q = tok.token_to_id(Q)
    segs = []
    for i, cap in enumerate(caps):
        if i:
            sep = tok.encode(separator(arg_names[i]))
            if sep[0] != q or sep[-1] != q:
                raise ValueError(f"separator does not tokenize around the delimiter token: {sep}")
            segs.append(Force(sep))
        segs.append(Generate(cap, [q]))
    res = be.decode_segments(tok.encode(text), segs, draws=draws, seeds=seeds, sampling=sampling)
    out = []
    for r in res:
        gens = [s for j, s in enumerate(r) if j % 2 == 0]  # Generate segments sit at even positions
        out.append([("", 0, 0) if s.skipped else (tok.decode(s.tokens), s.closed, len(s.tokens)) for s in gens])
    return out


def fullcall_gen(be, call: dict, draws: int, seeds: Sequence[int], prompt_key: str = "prompt",
                 sampling: Sampling = SERVED) -> list[dict]:
    tok = be.tokenizer
    text = call[prompt_key] + split_prefix(call)
    cap_new = min(2048, 2 * len(tok.encode(call.get("new_string") or "")) + 64)
    cap_old = min(1024, 2 * len(tok.encode(call["old_string"])) + 64)
    res = regenerate_args(be, text, ["new_string", "old_string"], [cap_new, cap_old], draws, seeds, sampling)
    n_prompt = len(tok.encode(text))
    out = []
    for d, ((new, closed_new, n_new), (old, closed, n_old)) in enumerate(res):
        out.append({"key": call["key"], "draw": d, "gen_new": new, "gen": old, "closed_new": closed_new, "closed": closed,
                    "n_gen_new": n_new, "n_gen": n_old, "prompt_tokens": n_prompt})
    return out


def run_replay(be, calls: Sequence[dict], mode: str = "frozen", draws: int = 2, seed_policy: str = "index",
               edit_set=None, edit_label: str | None = None, out: str | Path | None = None,
               prompt_key: str = "prompt", progress: Callable[[int, int, dict], None] | None = None) -> list[dict]:
    """Generate every (call, draw) under one arm. Resumable: (key, draw) pairs already in `out` are skipped.
    Each row records `edit` = edit_label (the gate checks it names the arm's frozen edit directory)."""
    fn = {"frozen": frozen_gen, "fullcall": fullcall_gen}[mode]
    done, rows = set(), []
    out_p = Path(out) if out else None
    if out_p and out_p.exists():
        for line in out_p.open():
            r = json.loads(line)
            done.add((r["key"], r["draw"]))
            rows.append(r)
    fh = None
    if out_p:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fh = out_p.open("a")
    try:
        with be.edits(edit_set):
            for i, c in enumerate(calls):
                if all((c["key"], d) in done for d in range(draws)):
                    continue
                seeds = seeds_for(seed_policy, c["key"], i, draws)
                for g in fn(be, c, draws, seeds, prompt_key=prompt_key):
                    if (g["key"], g["draw"]) in done:
                        continue
                    g["edit"] = edit_label
                    g["backend"] = be.name
                    rows.append(g)
                    if fh:
                        fh.write(json.dumps(g) + "\n")
                        fh.flush()
                if progress:
                    progress(i, len(calls), c)
    finally:
        if fh:
            fh.close()
    return rows


def score_rows(gens: Iterable[dict], calls: dict[str, dict], scorer, mode: str = "frozen") -> list[dict]:
    """Score generation rows with an EditScorer; `calls` maps key -> call record."""
    out = []
    for g in gens:
        c = calls[g["key"]]
        if mode == "frozen":
            row = scorer.score(c, g["gen"], closed=g.get("closed", 1))
        else:
            row = scorer.score_fullcall(c, g["gen"], g["gen_new"], closed=g.get("closed", 1), closed_new=g.get("closed_new", 1))
        row["draw"] = g["draw"]
        out.append(row)
    return out
