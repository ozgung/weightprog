"""Amended compile guard on a full-call replay (generalised from the study's `gate_r2_fullcall.py`).

Why it exists (the frozen-prefix compile trap). Gemma 4 writes a tool call's arguments in sorted key order, so for
`edit_file` it writes new_string BEFORE old_string. A frozen-prefix replay keeps the served model's new_string in
the prompt and regenerates only old_string. When an edit turns an escaped old_string into an exact copy, the harness
stops taking its unescape fallback (which also unescapes new_string) and inserts the frozen, still-escaped new_string
verbatim, so the file stops compiling whatever the edited model would have written. In the study every
artefact-reducing pass of all three proposers raised the frozen-prefix compile-fail rate (+2 to +21 points).

The fix: regenerate both arguments under the same arm (`replay.fullcall_replay`), score the generated pair, and
apply the same rule (edit compile-fail rate among accepted <= base + margin) to those rows. Everything else in the
registered rule stays as registered: amended primary pass = registered primary pass with the compile guard replaced.
"""

from __future__ import annotations

from typing import Mapping

from ..stats import holm
from .analysis import GateBlocked, GateSpec

AMENDMENT_TEXT = "full-call compile guard, 2026-11-05 (see weightprog.gate.fullcall)"


def summary(arm: Mapping, keys) -> dict:
    acc = [arm[k] for k in keys if arm[k]["accepted"]]
    n = len(keys)
    out = {"n": n, "accepted": len(acc) / n,
           "compile_fail_rate_among_accepted": (sum(1 for r in acc if r["compiles"] == 0) / len(acc)) if acc else None,
           "accepted_and_compiling": sum(1 for r in acc if r["compiles"] == 1)}
    if all("compiles_frozen_new" in arm[k] for k in keys):
        out["compile_fail_rate_among_accepted_frozen_new"] = (sum(1 for r in acc if r["compiles_frozen_new"] == 0) / len(acc)) if acc else None
    for f, name in (("artefact_old", "artefact_old"), ("artefact_new", "artefact_new"), ("exact_copy", "exact_copy")):
        if all(f in arm[k] for k in keys):
            out[name] = sum(arm[k][f] for k in keys) / n
    if all("closed_new" in arm[k] for k in keys):
        out["new_unclosed"] = sum(1 - arm[k]["closed_new"] for k in keys) / n
    return out


def check_universe(arm_name: str, rows: Mapping, universe: set) -> None:
    ks = set(rows)
    if ks != universe:
        raise GateBlocked(f"{arm_name}: {len(ks)} records, universe {len(universe)}, missing {len(universe - ks)}, "
                          f"extra {len(ks - universe)}")


def amend(registered: dict, base: Mapping, candidates: Mapping[str, Mapping], keys, spec: GateSpec | None = None,
          registered_name: str = "", strict: bool = True) -> dict:
    """registered: the registered report (analysis.analyze output). base / candidates: full-call scored rows keyed
    by (key, draw). strict=True BLOCKs unless the candidate set equals the registered one."""
    s = spec or GateSpec()
    keys = sorted(keys)
    uni = set(keys)
    check_universe("base", base, uni)
    cands = list(candidates)
    if strict and set(cands) != set(registered["candidates"]):
        raise GateBlocked(f"candidates {sorted(cands)} differ from the registered report's {sorted(registered['candidates'])}")
    m = registered["holm_m"]
    S = f"secondary_{s.secondary_name}"
    out = {"amendment": AMENDMENT_TEXT, "registered_report": registered_name, "holm_m": m, "base": summary(base, keys),
           "candidates": {}}
    margin = s.compile_margin if s.compile_margin is not None else 0.02
    for c in cands:
        r = registered["candidates"][c]
        e = candidates[c]
        check_universe(c, e, uni)
        se = summary(e, keys)
        cb, ce = out["base"]["compile_fail_rate_among_accepted"] or 0, se["compile_fail_rate_among_accepted"] or 0
        both = [k for k in keys if base[k]["accepted"] and e[k]["accepted"]]
        res = {"fullcall": se, "fullcall_compile_guard_pass": ce <= cb + margin,
               "paired_accepted_and_compiling": {
                   "both": sum(1 for k in both if base[k]["compiles"] == 1 and e[k]["compiles"] == 1),
                   "only_base": sum(1 for k in keys if base[k]["accepted"] and base[k]["compiles"] == 1
                                    and not (e[k]["accepted"] and e[k]["compiles"] == 1)),
                   "only_edit": sum(1 for k in keys if e[k]["accepted"] and e[k]["compiles"] == 1
                                    and not (base[k]["accepted"] and base[k]["compiles"] == 1))},
               "registered": {x: r[x] for x in ("primary_pass", "secondary_pass", "compile_guard_pass",
                                                f"{s.noninferiority_name}_noninferior", f"{s.preservation_name}_rule_pass",
                                                "primary_holm_pass") if x in r}}
        dirkey = "negative_means" if s.primary_lower_better else "positive_means"
        res["amended_primary_pass"] = (r["primary_holm_pass"] and r[f"primary_{s.primary_name}"][dirkey]
                                       and r[f"{s.noninferiority_name}_noninferior"] and r[f"{s.preservation_name}_rule_pass"]
                                       and res["fullcall_compile_guard_pass"])
        res["secondary_p"] = r[S]["p_raw"] if (res["amended_primary_pass"] and S in r) else 1.0
        out["candidates"][c] = res
    hs = holm({c: out["candidates"][c]["secondary_p"] for c in cands}, m, s.alpha)
    sec_dir = "negative_means" if s.secondary_lower_better else "positive_means"
    for c in cands:
        x = out["candidates"][c]
        x["secondary_holm_threshold"], x["secondary_holm_pass"] = hs[c]
        x["amended_secondary_pass"] = (x["amended_primary_pass"] and x["secondary_holm_pass"]
                                       and registered["candidates"][c].get(S, {}).get(sec_dir, False))
    return out
