"""Run cases under the base model, your edit and matched random edits, and say whether the edit did its job.

    from weightprog.evaluate import open_model, run, side_effects, compare, Criteria
    be = open_model(model_dir)                                    # MLX on Apple silicon, else PyTorch
    cases = load_cases("cases.jsonl")
    arms = {"base": None, "edit": EditSet.load_dir("edit/"),
            "rand1": matched_control(edit, seed=1), "rand2": matched_control(edit, seed=2)}
    rows = run(be, select(cases, split="tune"), arms, draws=2, model_dir=model_dir)
    side = side_effects(be, arms)                                 # KL / NLL on reference text and code
    report = compare(rows, side, arm="edit", controls=["rand1", "rand2"], criteria=Criteria())
    print("\\n".join(report["summary"]))

What `compare` checks (all thresholds live in `Criteria`; fix them before you look at holdout results):

1. **Target.** On `target` cases the edit must beat the base model *and every matched random control*: a one-sided
   exact sign test over cases (case-level improvement: lower bad rate for judged cases, higher log-prob margin for
   contrast cases) at `alpha`; the group-clustered bootstrap CI of the improvement over base must lie above 0; and
   for judged cases the good rate must rise by at least `min_good_share` of the drop in the bad rate (turning bad
   outputs into garbage is not a fix). Judged and contrast targets are tested separately. A random edit of the same
   modules, rank and norm is the right null: if it moves the metric too, your edit's effect is not specific.
2. **Keep.** On `keep` cases at most `keep_max_broken` cases per draw may go from `good` under base to not-`good`
   under the edit (contrast keep cases: margin from > 0 to <= 0), and at most `keep_max_changed_frac` of "unchanged"
   keep cases may change their greedy text. The controls' own breakage is printed beside it as the noise floor
   (sampled draws flip under any perturbation).
3. **Side effects.** On reference text and code (built-in, or yours): mean KL(base || edit) per token at most
   `kl_max` nats and mean NLL increase at most `nll_max` nats per token. Matched random controls are reported beside
   the edit so you can see what "no specific change" looks like.

The verdict is PASS only when all three were measured and passed; FAIL when a measured check failed; INCOMPLETE when
keep cases, controls or side effects are missing. For quick iterations run base once and reuse its rows
(`run(be, cases, {"base": None})`, then `run(be, cases, {"edit": e})`, then `compare(base_rows + edit_rows, ...)`).

Sampling: draw 0 is greedy, draws >= 1 sample at the served settings (T 0.6, top-p 0.95, top-k 64). Seeds depend on
the case id and draw only, so every arm sees the same seeds and comparisons are paired.
"""

from __future__ import annotations

import collections
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .backends import SERVED, Sampling, get_backend, seed_of
from .stats import bootstrap_clustered_ci, mean, sign_test
from .target import Case, cut_at_stop, judge

STOP_TOKENS = ("<eos>", "<turn|>")


def open_model(model_dir: str | Path, backend: str = "auto", dtype: str = "bfloat16", device: str | None = None):
    """Load a Gemma 4 checkpoint directory with the best available backend (`auto`: MLX on Apple silicon)."""
    if backend == "auto":
        backend = "mlx" if platform.system() == "Darwin" and platform.machine() == "arm64" else "torch"
    return get_backend(backend, model_dir, dtype=dtype, device=device)


def stop_ids(tokenizer, extra: Iterable[str] = ()) -> list[int]:
    out = []
    for t in (*STOP_TOKENS, *extra):
        i = tokenizer.token_to_id(t)
        if i is not None:
            out.append(i)
    return out


_STOP_CACHE: dict[tuple[int, str], list[int]] = {}


def stop_string_ids(tokenizer, s: str) -> list[int]:
    """Ids of vocabulary tokens that decode to the stop string `s` (ignoring surrounding whitespace), e.g. "```"
    and " ```". Generation ends at such a token, so a stop string costs no extra decoding; strings that span several
    tokens are still cut after the fact."""
    key = (id(tokenizer), s)
    if key not in _STOP_CACHE:
        want = s.strip()
        _STOP_CACHE[key] = sorted(i for _, i in tokenizer.vocab().items()
                                  if want and tokenizer.decode([i]).strip() == want)
    return _STOP_CACHE[key]


def seq_logprob(backend, prompt_ids: Sequence[int], cont_ids: Sequence[int]) -> float:
    """log P(continuation | prompt) for given token ids (sum over the continuation's tokens; longer continuations
    get lower totals, so compare continuations of similar token length, or single decision tokens)."""
    if not cont_ids:
        raise ValueError("empty continuation")
    ids = list(prompt_ids) + list(cont_ids)
    pos = list(range(len(prompt_ids) - 1, len(ids) - 1))
    lp = backend.logprobs(ids, pos)
    return float(sum(lp[j, t] for j, t in enumerate(cont_ids)))


def _cont_ids(tok, prompt_text: str, cont: str, prompt_ids: Sequence[int]) -> list[int]:
    """The continuation's tokens as the model would see them after the prompt: tokenise prompt + continuation together
    and keep what follows the prompt's own tokens (falls back to tokenising the continuation alone if a merge crosses
    the boundary)."""
    full = tok.encode(prompt_text + cont)
    n = len(prompt_ids)
    return full[n:] if full[:n] == list(prompt_ids) and len(full) > n else tok.encode(cont)


def case_fingerprint(case: Case, text: str, draws: int, sampling: Sampling, seed: str) -> str:
    """What a row depends on besides the arm: prompt text, rule, length, stops, draws, sampling and seed."""
    blob = json.dumps([text, case.good, case.bad, case.check, case.contrast, case.max_new, case.stop, draws,
                       [sampling.temperature, sampling.top_p, sampling.top_k], seed], ensure_ascii=False)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def reuse_rows(cached: Sequence[dict], cases: Sequence[Case], arm: str, draws: int = 2, sampling: Sampling = SERVED,
               seed: str = "weightprog", model_dir=None) -> tuple[list[dict], list[Case]]:
    """Split `cases` into (rows of `arm` from `cached` whose fingerprint still matches, cases that must be rerun)."""
    have = collections.defaultdict(list)
    for r in cached:
        if r["arm"] == arm:
            have[(r["id"], r.get("case_sha"))].append(r)
    keep, todo = [], []
    for c in cases:
        rs = have.get((c.id, case_fingerprint(c, c.text(model_dir), draws, sampling, seed)))
        n = 1 if c.mode == "contrast" else draws
        if rs and len(rs) == n:
            keep += rs
        else:
            todo.append(c)
    return keep, todo


def run(backend, cases: Sequence[Case], arms: dict | None = None, draws: int = 2, sampling: Sampling = SERVED,
        seed: str = "weightprog", model_dir: str | Path | None = None,
        progress: Callable[[str], None] | None = None) -> list[dict]:
    """Generate (or score, for contrast cases) every case under every arm. Returns one row per (arm, case, draw).

    `arms` maps a name to an EditSet or None (None = the unedited model). Default: {"base": None}, which is a census
    of the base model's behaviour."""
    arms = arms if arms is not None else {"base": None}
    tok = backend.tokenizer
    stops = stop_ids(tok)
    texts = {c.id: c.text(model_dir) for c in cases}
    rows = []
    for arm, edit in arms.items():
        with backend.edits(edit):
            for k, c in enumerate(cases):
                ids = tok.encode(texts[c.id])
                base = {"arm": arm, "id": c.id, "role": c.role, "split": c.split, "group": c.group, "mode": c.mode,
                        "case_sha": case_fingerprint(c, texts[c.id], draws, sampling, seed)}
                if c.mode == "contrast":
                    lg = seq_logprob(backend, ids, _cont_ids(tok, texts[c.id], c.contrast["good"], ids))
                    lb = seq_logprob(backend, ids, _cont_ids(tok, texts[c.id], c.contrast["bad"], ids))
                    rows.append({**base, "draw": 0, "margin": lg - lb, "lp_good": lg, "lp_bad": lb})
                else:
                    seeds = [seed_of(f"{seed}|{c.id}", d) for d in range(draws)]
                    st = stops + [i for s in c.stop for i in stop_string_ids(tok, s)]
                    outs = backend.generate(ids, c.max_new, draws=draws, seeds=seeds, stop_ids=st,
                                            sampling=sampling)
                    for d, o in enumerate(outs):
                        text = cut_at_stop(tok.decode(o.tokens), c.stop)
                        rows.append({**base, "draw": d, "text": text, "n_tokens": len(o.tokens),
                                     "label": judge(text, c) if c.mode == "judged" else None})
                if progress:
                    progress(f"[{arm}] {k + 1}/{len(cases)} {c.id}")
    return rows


# ---- reference text for side effects ------------------------------------------------------------------------

def reference_texts(kind: str = "fit") -> list[str]:
    """Built-in original prose, code and chat text (Apache-2.0, written for this library).

    kind="fit" (16 texts): use these for gate negatives and "ordinary" rows. kind="eval" (24 texts, disjoint, with
    chat-formatted turns): `side_effects` uses these by default, so the side-effect check is not in-sample for a gate
    fitted on the fit set."""
    d = json.loads(resources.files("weightprog").joinpath("resources/reference.json").read_text())
    if kind not in ("fit", "eval"):
        raise ValueError("kind is 'fit' or 'eval'")
    return d[kind]


def side_effects(backend, arms: dict, texts: Sequence[str] | None = None, chunk: int = 64,
                 bos: str = "<bos>") -> dict:
    """Per arm, against the unedited model, over every position of every reference text (default: the built-in
    `reference_texts("eval")`): mean and 95th-percentile KL(base || arm) in nats, top-1 agreement, and the mean change
    of the per-token NLL of the actual text. Each text is prefixed with `bos`."""
    texts = list(texts) if texts is not None else reference_texts("eval")
    tok = backend.tokenizer
    b = tok.token_to_id(bos)
    acc = {a: {"kl": [], "agree": [], "dnll": []} for a, e in arms.items() if e is not None}
    for t in texts:
        ids = ([b] if b is not None else []) + tok.encode(t)
        if len(ids) < 2:
            continue
        positions = list(range(0, len(ids) - 1))
        for i in range(0, len(positions), chunk):
            pos = positions[i:i + chunk]
            nxt = [ids[p + 1] for p in pos]
            lp0 = backend.logprobs(ids, pos).astype(np.float64)
            p0 = np.exp(lp0)
            top0 = lp0.argmax(-1)
            nll0 = -lp0[np.arange(len(pos)), nxt]
            for a, e in arms.items():
                if e is None:
                    continue
                with backend.edits(e):
                    lp1 = backend.logprobs(ids, pos).astype(np.float64)
                acc[a]["kl"] += list((p0 * (lp0 - lp1)).sum(-1))
                acc[a]["agree"] += list(top0 == lp1.argmax(-1))
                acc[a]["dnll"] += list(-lp1[np.arange(len(pos)), nxt] - nll0)
    out = {}
    for a, v in acc.items():
        kl = np.array(v["kl"])
        out[a] = {"kl_mean": float(kl.mean()), "kl_p95": float(np.percentile(kl, 95)),
                  "top1_agree": float(np.mean(v["agree"])), "nll_delta": float(np.mean(v["dnll"])),
                  "n_positions": int(len(kl)), "n_texts": len(texts)}
    return out


# ---- judgement ---------------------------------------------------------------------------------------------

@dataclass
class Criteria:
    alpha: float = 0.05                # one-sided sign test over target cases: edit better than base and every control
    require_group_ci: bool = True      # also: group-clustered bootstrap CI of the improvement vs base above 0
    min_good_share: float = 0.5        # judged targets: good-rate gain vs base >= this share of the bad-rate drop
                                       # (an edit that turns bad outputs into garbage, 'other', is not a fix)
    keep_max_broken: int = 2           # keep cases per draw that go from good (base) to not good (edit)
    keep_max_changed_frac: float = 0.10  # "unchanged" keep cases whose greedy text changed
    kl_max: float = 0.05               # mean KL(base || edit), nats per token, reference texts
    nll_max: float = 0.02              # mean NLL increase, nats per token, reference texts


def _index(rows):
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["arm"], r["id"])][r["draw"]] = r
    return by


def case_scores(rows: Sequence[dict], arm: str) -> dict[str, dict]:
    """Per case under one arm: bad / good / other rates over draws (judged), margin (contrast), greedy text."""
    out = {}
    for (a, cid), draws in _index(rows).items():
        if a != arm:
            continue
        rs = [draws[d] for d in sorted(draws)]
        r0 = rs[0]
        s = {"role": r0["role"], "mode": r0["mode"], "group": r0["group"], "split": r0["split"]}
        if r0["mode"] == "contrast":
            s["margin"] = r0["margin"]
        else:
            s["greedy"] = r0["text"]
            if r0["mode"] == "judged":
                labels = [r["label"] for r in rs]
                for lab in ("bad", "good", "other"):
                    s[lab] = labels.count(lab) / len(labels)
                s["labels"] = labels
        out[cid] = s
    return out


def improvement(arm_s: dict, ref_s: dict) -> float:
    """Positive = the arm is better on this target case (lower bad rate, or larger margin)."""
    if arm_s["mode"] == "contrast":
        return arm_s["margin"] - ref_s["margin"]
    return ref_s["bad"] - arm_s["bad"]


def _target_test(sa, sr, ids, n_boot=10000):
    u = {c: improvement(sa[c], sr[c]) for c in ids}
    p, n_better, n_worse = sign_test(u.values())
    ci = bootstrap_clustered_ci(u, {c: sa[c]["group"] for c in ids}, n=n_boot) if len(ids) > 1 else [None, None]
    out = {"p_one_sided": p, "cases_better": n_better, "cases_worse": n_worse, "cases": len(ids),
           "mean_improvement": mean(list(u.values())), "ci95_group_bootstrap": ci,
           "groups": len({sa[c]["group"] for c in ids})}
    if sa[ids[0]]["mode"] == "judged":
        out["mean_good_change"] = mean([sa[c]["good"] - sr[c]["good"] for c in ids])
        out["bad_to_other_cases"] = sum(sa[c]["bad"] < sr[c]["bad"] and sa[c]["other"] > sr[c]["other"] for c in ids)
    return out


def _keep_changes(by, sb, sa, keep, draws, base, arm):
    broken = {d: [] for d in draws}
    fixed = {d: [] for d in draws}
    changed, n_unch = [], 0
    for c in keep:
        mode = sb[c]["mode"]
        if mode == "unchanged":
            n_unch += 1
            if sa[c]["greedy"] != sb[c]["greedy"]:
                changed.append(c)
        elif mode == "contrast":
            if sb[c]["margin"] > 0 >= sa[c]["margin"]:
                broken[draws[0]].append(c)
            elif sa[c]["margin"] > 0 >= sb[c]["margin"]:
                fixed[draws[0]].append(c)
        else:
            for d in draws:
                lb, la = by[(base, c)][d]["label"], by[(arm, c)][d]["label"]
                if lb == "good" and la != "good":
                    broken[d].append(c)
                elif la == "good" and lb != "good":
                    fixed[d].append(c)
    return broken, fixed, changed, n_unch


def compare(rows: Sequence[dict], side: dict | None = None, arm: str = "edit", base: str = "base",
            controls: Sequence[str] = (), criteria: Criteria | None = None, n_boot: int = 10000) -> dict:
    """Judge `arm` against `base` and each control on the rows of `run` (and the output of `side_effects`).

    Verdict: PASS (every check measured and passed), FAIL (a measured check failed) or INCOMPLETE (no failure, but
    keep cases, matched controls or side effects were not measured, so a pass would not mean anything)."""
    cr = criteria or Criteria()
    by = _index(rows)
    sb, sa = case_scores(rows, base), case_scores(rows, arm)
    sc = {k: case_scores(rows, k) for k in controls}
    if set(sa) != set(sb) or any(set(v) != set(sb) for v in sc.values()):
        raise ValueError("arms do not cover the same cases")
    rep: dict = {"arm": arm, "base": base, "controls": list(controls), "criteria": asdict(cr), "n_cases": len(sb)}

    # rates of the base model and every arm by role (judged cases)
    census = {}
    for name, s in (("base", sb), (arm, sa), *sc.items()):
        for role in ("target", "keep"):
            js = [v for v in s.values() if v["role"] == role and v["mode"] == "judged"]
            if js:
                census.setdefault(name, {})[role] = {lab: mean([v[lab] for v in js]) for lab in ("bad", "good", "other")}
    rep["rates"] = census

    # 1. target: each mode present (judged, contrast) is tested on its own and must pass
    tgt = [c for c, v in sb.items() if v["role"] == "target"]
    target = {}
    for mode in ("judged", "contrast"):
        ids = [c for c in tgt if sb[c]["mode"] == mode]
        if not ids:
            continue
        tests = {base: _target_test(sa, sb, ids, n_boot)}
        for k in controls:
            tests[k] = _target_test(sa, sc[k], ids, n_boot)
        for k in controls:  # does the control itself move the metric? (it should not)
            tests[f"{k}_vs_{base}"] = _target_test(sc[k], sb, ids, n_boot)
        ok = all(tests[r]["p_one_sided"] < cr.alpha and tests[r]["mean_improvement"] > 0 for r in (base, *controls))
        ci = tests[base]["ci95_group_bootstrap"]
        if cr.require_group_ci:
            ok = ok and ci[0] is not None and ci[0] > 0
        if mode == "judged" and cr.min_good_share > 0:
            ok = ok and tests[base]["mean_good_change"] >= cr.min_good_share * tests[base]["mean_improvement"]
        target[mode] = {"metric": "bad rate (lower is better)" if mode == "judged" else "log-prob margin (higher is better)",
                        "tests": tests, "pass": bool(ok)}
    target_pass = bool(target) and all(v["pass"] for v in target.values())
    rep["target"] = {"by_mode": target, "pass": target_pass}

    # 2. keep (the controls' own breakage is the noise floor to read the edit's against)
    keep = [c for c, v in sb.items() if v["role"] == "keep"]
    draws = sorted({d for (a, c), ds in by.items() if a == base for d in ds})
    broken, fixed, changed, n_unch = _keep_changes(by, sb, sa, keep, draws, base, arm)
    max_broken = max((len(v) for v in broken.values()), default=0)
    frac = len(changed) / n_unch if n_unch else 0.0
    keep_pass = max_broken <= cr.keep_max_broken and frac <= cr.keep_max_changed_frac
    ctrl = {}
    for k in controls:
        b2, f2, c2, _ = _keep_changes(by, sb, sc[k], keep, draws, base, k)
        ctrl[k] = {"broken_per_draw": {str(d): len(v) for d, v in b2.items()},
                   "fixed_per_draw": {str(d): len(v) for d, v in f2.items()}, "changed": len(c2)}
    rep["keep"] = {"cases": len(keep), "broken_per_draw": {str(d): v for d, v in broken.items()},
                   "fixed_per_draw": {str(d): v for d, v in fixed.items()},
                   "unchanged_cases": n_unch, "changed": changed, "changed_frac": frac, "controls": ctrl,
                   "pass": keep_pass}

    # 3. side effects
    if side:
        s = side.get(arm)
        side_pass = bool(s) and s["kl_mean"] <= cr.kl_max and s["nll_delta"] <= cr.nll_max
        rep["side_effects"] = {"by_arm": side, "pass": side_pass}
    else:
        side_pass = None
        rep["side_effects"] = {"by_arm": None, "pass": None, "note": "not measured"}

    missing = [m for m, absent in (("keep cases", not keep), ("matched controls", not controls),
                                   ("side effects", side_pass is None)) if absent]
    failed = not target_pass or not keep_pass or side_pass is False
    rep["verdict"] = "FAIL" if failed else ("INCOMPLETE" if missing else "PASS")
    rep["missing"] = missing
    rep["pass"] = rep["verdict"] == "PASS"
    rep["summary"] = summary_lines(rep)
    return rep


def summary_lines(rep: dict) -> list[str]:
    a, b = rep["arm"], rep["base"]
    L = [f"== {a} vs {b}" + (f" and {', '.join(rep['controls'])}" if rep["controls"] else "") + f" ({rep['n_cases']} cases)"]
    for name, by_role in rep["rates"].items():
        for role, r in by_role.items():
            L.append(f"  {name:>8} {role:6} bad {r['bad']:.3f}  good {r['good']:.3f}  other {r['other']:.3f}")
    if not rep["target"]["by_mode"]:
        L.append("target: no target cases: fail")
    for mode, t in rep["target"]["by_mode"].items():
        L.append(f"target [{mode}] ({t['metric']}): {'PASS' if t['pass'] else 'fail'}")
        for ref, s in t["tests"].items():
            ci = s["ci95_group_bootstrap"]
            cis = f" group CI [{ci[0]:+.3f}, {ci[1]:+.3f}] ({s['groups']} groups)" if ci[0] is not None else ""
            extra = (f", good {s['mean_good_change']:+.3f}, bad->other {s['bad_to_other_cases']}"
                     if "mean_good_change" in s else "")
            L.append(f"  vs {ref:>14}: better {s['cases_better']} / worse {s['cases_worse']} of {s['cases']}, "
                     f"p {s['p_one_sided']:.3g}, mean improvement {s['mean_improvement']:+.3f}{cis}{extra}")
    k = rep["keep"]
    L.append(f"keep: {'PASS' if k['pass'] else 'fail'}  broken per draw "
             f"{ {d: len(v) for d, v in k['broken_per_draw'].items()} }, fixed "
             f"{ {d: len(v) for d, v in k['fixed_per_draw'].items()} }, unchanged-changed "
             f"{len(k['changed'])}/{k['unchanged_cases']}")
    for name, v in k["controls"].items():
        L.append(f"  {name:>8} (noise floor): broken per draw {v['broken_per_draw']}, fixed {v['fixed_per_draw']}, "
                 f"unchanged-changed {v['changed']}")
    s = rep["side_effects"]
    if s["by_arm"]:
        L.append(f"side effects: {'PASS' if s['pass'] else 'fail'}")
        for name, v in s["by_arm"].items():
            L.append(f"  {name:>8}: KL mean {v['kl_mean']:.4f} p95 {v['kl_p95']:.4f}  top-1 agree {v['top1_agree']:.4f}"
                     f"  NLL {v['nll_delta']:+.4f}  ({v['n_positions']} positions)")
    else:
        L.append("side effects: not measured")
    L.append(f"VERDICT: {rep['verdict']}" + (f" (not measured: {', '.join(rep['missing'])})" if rep["missing"] else ""))
    return L


def write_rows(rows: Sequence[dict], path: str | Path) -> None:
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def read_rows(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
