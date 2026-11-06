"""Reading gate inputs with the completeness checks that make the gate refuse (BLOCK) instead of guessing.

Layout (the study's, and what `weightprog gate` expects):

    GATE_DIR/gen_<arm>.jsonl          one generation per (key, draw): {"key", "draw", "gen", "edit", ...}
    GATE_DIR/gen_<arm>.scored.jsonl   one scored row per (key, draw): {"key", "draw", "accepted", "recorded_kind",
                                      "compiles", "exact_copy", ...}
    GATE_DIR/fullcall/gen_<arm>[.scored].jsonl   the same for the full-call replay (both arguments regenerated)

BLOCK conditions: duplicate (key, draw) records; an arm missing or having extra records in generation or scoring;
an edit arm whose records do not name its own edit directory (`.../<arm>/edit`); a base record that has an edit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from .analysis import GateBlocked


def read_jsonl_unique(p: str | Path) -> dict:
    out = {}
    p = Path(p)
    if not p.exists():
        raise GateBlocked(f"{p} missing")
    with p.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r["key"], r["draw"])
            if k in out:
                raise GateBlocked(f"duplicate {k} in {p.name}")
            out[k] = r
    return out


def universe(keys_file: str | Path, draws: Iterable[int] = (0, 1)) -> list[tuple[str, int]]:
    want = [l.strip() for l in Path(keys_file).read_text().splitlines() if l.strip()]
    if len(set(want)) != len(want):
        raise GateBlocked("duplicate keys in the key file")
    return sorted((k, d) for k in want for d in draws)


def edit_names_arm(edit_field, arm: str) -> bool:
    return bool(edit_field) and Path(edit_field).parent.name == arm


def load_arm(gate_dir: str | Path, arm: str, keys, edit_expected: bool, metrics: dict[str, Callable] | None = None,
             sub: str = "") -> dict:
    """Scored rows of one arm keyed by (key, draw), after the completeness checks. `metrics` adds per-row fields
    computed from the generation record: {name: fn(gen_record) -> int} (e.g. the escape-artefact metric)."""
    g_dir = Path(gate_dir) / sub if sub else Path(gate_dir)
    g, s = read_jsonl_unique(g_dir / f"gen_{arm}.jsonl"), read_jsonl_unique(g_dir / f"gen_{arm}.scored.jsonl")
    ks = set(keys)
    for name, d in (("gen", g), ("scored", s)):
        miss, extra = ks - set(d), set(d) - ks
        if miss or extra:
            raise GateBlocked(f"{arm} {name}: missing {len(miss)} / extra {len(extra)}")
    for k in keys:
        e = g[k].get("edit")
        if edit_expected and not edit_names_arm(e, arm):
            raise GateBlocked(f"{arm}: record {k} not generated with {arm}/edit (edit={e})")
        if not edit_expected and e:
            raise GateBlocked(f"base record {k} has an edit")
        for name, fn in (metrics or {}).items():
            s[k][name] = fn(g[k])
    return s


def load_study_calls(replay_dir: str | Path) -> dict[str, dict]:
    """calls.jsonl of a replay directory, keyed by call key (prompts included; large)."""
    with (Path(replay_dir) / "calls.jsonl").open() as f:
        return {r["key"]: r for r in map(json.loads, f)}
