"""The verification gate: paired task-macro sign tests of a frozen edit against base and matched random controls.

Generalised from the study's round-2 analysis (`gate_r2_analyze.py`), whose rule this module implements with every
threshold a parameter (`GateSpec`). With the default spec it reproduces the registered round-2 report exactly.

Inputs are per-arm dicts {(call key, draw): record}, where a record holds 0/1 metric fields (by default `accepted`,
`escape_artefact`), `recorded_kind` (the call's outcome when it was served) and `compiles` (1, 0 or None). The unit of
analysis is the task: `unit_of(call key)`; `cluster_of(task)` gives the repository for the clustered bootstrap.

Rule (all from GateSpec):
1. PRIMARY. Per task, mean of the primary metric over its calls x draws; paired per-task differences edit - ref for
   ref in {base, each control}: a mean in the wanted direction AND a one-sided exact sign test p < alpha for every
   ref. p_primary = max over the refs; Holm across the family (size holm_m, absent members hold p = 1 slots).
2. GUARDS (all required): non-inferiority of a task-macro metric vs base (diff >= -margin); preservation per draw
   (calls of a given recorded kind that base gets right and the edit gets wrong <= max(min, floor(frac * n)));
   compile-fail rate among accepted edits <= base + margin. Primary pass = 1 and 2.
3. KEY SECONDARY, fixed sequence: tested only for primary passes (otherwise p = 1); the same three-way comparison
   on the secondary metric; Holm over the same family size.
Reported, not part of the rule: task and cluster bootstrap CIs (vs base), per-source and per-recorded-kind tables,
and what the edit did to the base arm's primary-positive generations.
"""

from __future__ import annotations

import collections
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from ..stats import bootstrap_clustered_ci, bootstrap_mean_ci, holm, sign_test

Key = tuple  # (call key, draw)
Arm = Mapping[Key, dict]


class GateBlocked(SystemExit):
    """Raised when the inputs are incomplete or inconsistent: the gate refuses to produce a report."""

    def __init__(self, msg: str):
        super().__init__(f"BLOCKED: {msg}")
        self.msg = msg


def default_unit(call_key: str) -> str:
    """Task id of a replay key `run|task|item|tc` (the study's format); the whole key otherwise."""
    parts = call_key.split("|")
    return parts[1] if len(parts) > 1 else call_key


@dataclass
class GateSpec:
    primary_metric: str = "escape_artefact"
    primary_lower_better: bool = True
    secondary_metric: str | None = "accepted"
    secondary_lower_better: bool = False
    alpha: float = 0.05
    holm_m: int = 2
    controls: tuple[str, ...] = ("rand1", "rand2")
    noninferiority_metric: str = "accepted"
    noninferiority_margin: float = 0.01
    preservation_metric: str = "accepted"
    preservation_kind: str | None = "ok"          # recorded_kind of the calls whose base successes must be kept
    preservation_min: int = 2
    preservation_frac: float = 0.01
    compile_margin: float | None = 0.02
    draws: tuple[int, ...] = (0, 1)
    bootstrap_n: int = 10000
    bootstrap_seed: int = 20261104
    sources: dict[str, str] = field(default_factory=lambda: {"OOD-43": "o43", "OOD-B": "oodb"})  # name -> key prefix
    # report key names (defaults give the round-2 report's layout)
    primary_name: str = "escape_artefact"
    primary_short: str = "artefact"
    secondary_name: str = "acceptance"
    noninferiority_name: str = "acceptance"
    preservation_name: str = "served_ok"

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "GateSpec":
        d = dict(d)
        for k in ("controls", "draws"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


# ---- building blocks -------------------------------------------------------------------------------------------

def task_means(arm: Arm, keys, metric: str, unit_of: Callable[[str], str] = default_unit) -> dict[str, float]:
    by = collections.defaultdict(list)
    for k in keys:
        by[unit_of(k[0])].append(int(bool(arm[k][metric])))
    return {t: sum(v) / len(v) for t, v in by.items()}


def compare(e: Arm, r: Arm, keys, metric: str, lower_better: bool = False, alpha: float = 0.05,
            unit_of: Callable[[str], str] = default_unit) -> dict:
    """Paired task-macro comparison of arm e against reference r on one 0/1 metric."""
    te, tr = task_means(e, keys, metric, unit_of), task_means(r, keys, metric, unit_of)
    diffs = [te[t] - tr[t] for t in sorted(te)]
    p, better, worse = sign_test([-d for d in diffs] if lower_better else diffs)
    mean = sum(diffs) / len(diffs)
    return {"metric": metric, "task_macro_mean_diff": mean, "tasks_better": better, "tasks_worse": worse,
            "tasks_zero": len(diffs) - better - worse, "p_sign_one_sided": p,
            "pass": ((mean < 0) if lower_better else (mean > 0)) and p < alpha,
            "call_micro": [sum(int(bool(e[k][metric])) for k in keys), sum(int(bool(r[k][metric])) for k in keys), len(keys)]}


def boots(e: Arm, r: Arm, keys, cluster_of: Mapping[str, str], metric: str, n: int = 10000, seed: int = 20261104,
          unit_of: Callable[[str], str] = default_unit) -> dict:
    te, tr = task_means(e, keys, metric, unit_of), task_means(r, keys, metric, unit_of)
    d = {t: te[t] - tr[t] for t in te}
    return {"task_ci95": bootstrap_mean_ci(d, n, seed), "repo_clustered_ci95": bootstrap_clustered_ci(d, cluster_of, n, seed + 1),
            "n_tasks": len(d), "n_repos": len({cluster_of[t] for t in d})}


def compile_summary(arm: Arm, keys, accepted: str = "accepted") -> dict:
    acc = [arm[k] for k in keys if arm[k][accepted]]
    fail = sum(1 for r in acc if r["compiles"] == 0)
    return {"accepted": len(acc), "compile_fail_rate_among_accepted": fail / len(acc) if acc else None,
            "accepted_and_compiling": sum(1 for r in acc if r["compiles"] == 1),
            "unknown_compile": sum(1 for r in acc if r["compiles"] is None)}


def _dir(lower_better: bool) -> str:
    return "negative_means" if lower_better else "positive_means"


# ---- the gate ------------------------------------------------------------------------------------------------

def analyze(base: Arm, candidates: Mapping[str, Arm], controls: Mapping[str, Mapping[str, Arm]], keys,
            cluster_of: Mapping[str, str], spec: GateSpec | None = None,
            unit_of: Callable[[str], str] = default_unit) -> dict:
    """Run the registered rule. candidates: {name: arm}; controls: {name: {control name: arm}} with the control
    names of spec.controls. Returns the report dict (JSON-serialisable)."""
    s = spec or GateSpec()
    keys = sorted(keys)
    cands = list(candidates)
    refs = ("base",) + tuple(s.controls)
    pm, sm = s.primary_metric, s.secondary_metric
    rep = {"n_calls_x_draws": len(keys), "n_tasks": len({unit_of(k[0]) for k in keys}), "holm_m": max(s.holm_m, len(cands)),
           f"base_{pm}s": sum(base[k][pm] for k in keys), "candidates": {}}
    srcs = {name: [k for k in keys if k[0].startswith(prefix)] for name, prefix in s.sources.items()}
    for c in cands:
        e = candidates[c]
        ref = {"base": base, **{x: controls[c][x] for x in s.controls}}
        pri = {f"vs_{x}": compare(e, ref[x], keys, pm, s.primary_lower_better, s.alpha, unit_of) for x in refs}
        pri.update({f"{x}_vs_base": compare(ref[x], base, keys, pm, s.primary_lower_better, s.alpha, unit_of) for x in s.controls})
        pri["p"] = max(pri[f"vs_{x}"]["p_sign_one_sided"] for x in refs)
        pri[_dir(s.primary_lower_better)] = all(
            (pri[f"vs_{x}"]["task_macro_mean_diff"] < 0) if s.primary_lower_better else (pri[f"vs_{x}"]["task_macro_mean_diff"] > 0)
            for x in refs)
        res = {f"primary_{s.primary_name}": pri}
        if sm:
            sec = {f"vs_{x}": compare(e, ref[x], keys, sm, s.secondary_lower_better, s.alpha, unit_of) for x in refs}
            sec.update({f"{x}_vs_base": compare(ref[x], base, keys, sm, s.secondary_lower_better, s.alpha, unit_of) for x in s.controls})
            sec["p_raw"] = max(sec[f"vs_{x}"]["p_sign_one_sided"] for x in refs)
            sec[_dir(s.secondary_lower_better)] = all(
                (sec[f"vs_{x}"]["task_macro_mean_diff"] < 0) if s.secondary_lower_better else (sec[f"vs_{x}"]["task_macro_mean_diff"] > 0)
                for x in refs)
            res[f"secondary_{s.secondary_name}"] = sec
        # guard 1: non-inferiority vs base
        ni = compare(e, base, keys, s.noninferiority_metric, False, s.alpha, unit_of)
        res[f"{s.noninferiority_name}_noninferior"] = ni["task_macro_mean_diff"] >= -s.noninferiority_margin
        # guard 2: preservation of base successes, per draw
        loss = {}
        for d in s.draws:
            okb = [k for k in keys if k[1] == d and (s.preservation_kind is None or base[k]["recorded_kind"] == s.preservation_kind)
                   and base[k][s.preservation_metric]]
            lost = [k for k in okb if not e[k][s.preservation_metric]]
            lim = max(s.preservation_min, math.floor(s.preservation_frac * len(okb)))
            loss[f"draw{d}"] = {"n_lost": len(lost), "limit": lim, f"base_accepted_{s.preservation_name}": len(okb),
                                "tasks": sorted({unit_of(k[0]) for k in lost})}
        res[f"{s.preservation_name}_loss"] = loss
        res[f"{s.preservation_name}_rule_pass"] = all(v["n_lost"] <= v["limit"] for v in loss.values())
        # guard 3: compile
        if s.compile_margin is not None:
            cb, ce = compile_summary(base, keys), compile_summary(e, keys)
            res["compile"] = {"base": cb, "edit": ce}
            res["compile_guard_pass"] = (ce["compile_fail_rate_among_accepted"] or 0) <= (cb["compile_fail_rate_among_accepted"] or 0) + s.compile_margin
        else:
            res["compile_guard_pass"] = True
        base_pos = [k for k in keys if base[k][pm]]
        metrics = [pm] + ([sm] if sm else [])
        res["reported"] = {
            "bootstrap_vs_base": {mt: boots(e, base, keys, cluster_of, mt, s.bootstrap_n, s.bootstrap_seed, unit_of) for mt in metrics},
            "by_source": {sn: {mt: compare(e, base, ks, mt, (mt == pm and s.primary_lower_better) or (mt == sm and s.secondary_lower_better),
                                           s.alpha, unit_of) for mt in metrics} for sn, ks in srcs.items()},
            "by_recorded_kind": {kind: {**{mt: [sum(e[k][mt] for k in ks), sum(base[k][mt] for k in ks)] for mt in [sm or pm, pm]}, "n": len(ks)}
                                 for kind in sorted({base[k]["recorded_kind"] for k in keys})
                                 for ks in [[k for k in keys if base[k]["recorded_kind"] == kind]]},
            f"base_{s.primary_short}_generations_under_edit": {
                "n": len(base_pos), f"still_{s.primary_short}": sum(e[k][pm] for k in base_pos),
                "base_accepted": sum(base[k]["accepted"] for k in base_pos), "edit_accepted": sum(e[k]["accepted"] for k in base_pos)},
        }
        rep["candidates"][c] = res
    m = rep["holm_m"]
    P, S = f"primary_{s.primary_name}", f"secondary_{s.secondary_name}"
    hp = holm({c: rep["candidates"][c][P]["p"] for c in cands}, m, s.alpha)
    for c in cands:
        r = rep["candidates"][c]
        r["primary_holm_threshold"], r["primary_holm_pass"] = hp[c]
        r["primary_pass"] = (r["primary_holm_pass"] and r[P][_dir(s.primary_lower_better)]
                             and r[f"{s.noninferiority_name}_noninferior"] and r[f"{s.preservation_name}_rule_pass"]
                             and r["compile_guard_pass"])
        if sm:
            r[S]["p"] = r[S]["p_raw"] if r["primary_pass"] else 1.0
    if sm:
        hs = holm({c: rep["candidates"][c][S]["p"] for c in cands}, m, s.alpha)
        for c in cands:
            r = rep["candidates"][c]
            r["secondary_holm_threshold"], r["secondary_holm_pass"] = hs[c]
            r["secondary_pass"] = r["primary_pass"] and r["secondary_holm_pass"] and r[S][_dir(s.secondary_lower_better)]
    return rep


def verdict_lines(rep: dict, spec: GateSpec | None = None) -> list[str]:
    s = spec or GateSpec()
    P, S = f"primary_{s.primary_name}", f"secondary_{s.secondary_name}"
    out = []
    for c, r in rep["candidates"].items():
        pa = r[P]
        sa = r.get(S, {})
        line = (f"{c}: {s.primary_metric} vs base {pa['vs_base']['task_macro_mean_diff']:+.4f} (better {pa['vs_base']['tasks_better']} / "
                f"worse {pa['vs_base']['tasks_worse']}), p {pa['p']:.4g}")
        if sa:
            line += f"; {s.secondary_metric} vs base {sa['vs_base']['task_macro_mean_diff']:+.4f}"
        line += (f"; guards: noninf {r[f'{s.noninferiority_name}_noninferior']} {s.preservation_name} "
                 f"{r[f'{s.preservation_name}_rule_pass']} compile {r['compile_guard_pass']} -> PRIMARY {r['primary_pass']}")
        if "secondary_pass" in r:
            line += f", SECONDARY {r['secondary_pass']}"
        out.append(line)
    return out


def write_report(rep: dict, path: str | Path) -> Path:
    p = Path(path)
    p.write_text(json.dumps(rep, indent=1))
    return p
