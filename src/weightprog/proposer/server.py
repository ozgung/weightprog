"""Forward-pass budget server: one model process serving budgeted passes to proposer runs through a file queue.

    weightprog init-run --config server.json RUN_ID --proposer fable --model claude-fable-5-1 --effort max --budget 25
    weightprog serve --config server.json [--once]

Runs live in RUNS_DIR/<run_id>/ with state.json, requests/, edit/ (written by the kernel), work/, notebook.md.
A proposer writes requests/fwd_<k>.json; the server claims it (renames it to .json.claimed), runs it, writes
requests/fwd_<k>.done.json and counts one pass if it succeeded. A malformed, failed or refused request does not
count. Requests of a run are served in numeric order of k; runs in name order.

Request types (limits are frozen per server config; see LIMITS):
  Optional in any request: "expect_edit": {file name: sha256} pins the edit/ snapshot the pass must apply (the
  request fails, uncounted, on a mismatch); every .done.json records "edit_snapshot".
  Optional in gen and read: "rendering": "served" (default; the prompt as the agent saw it) | "raw" (the prompt
  from the call set's calls_raw.jsonl, when the server has one). The model's own prefix is the same in both.

  {"type": "gen", "arms": ["base", "edit"], "calls": [keys] | {"sample": N, "seed": S}, "draws": 1}
      Regenerate the last argument (old_string) of call-set calls (prompt + the model's own prefix), score each with
      the configured edit function on the true file state. Returns per arm: accepted / not_found / multiple rates,
      exact copies, escape artefacts, unclosed generations, compile failures among accepted edits, per recorded
      stratum, and paired base-vs-edit tables for acceptance and artefacts. Per-call rows (with generated text) in
      work/fwd_<k>/rows.jsonl. Base generations are cached (same seeds), so "base" is free after the first time.
      For runs initialised with --fullcall, gen also runs a full-call replay of the same calls and draws (unless the
      request says "fullcall": false): new_string then old_string from the same arm, the pair scored and compiled;
      returned under "fullcall" with "skipped" for calls whose prefix cannot be cut at new_string.
  {"type": "read", "arm": "base" | "edit", "texts": [{"key": K, "continuation": "recorded" | "<text>"}],
   "layers": [ints], "capture": ["resid", "mlp", "attn", "logprobs"], "attn_layers": [ints],
   "positions": "continuation" | "prompt_last"}
      Teacher-forced pass over prompt + prefix + continuation. Saves work/fwd_<k>/text_<i>.npz: resid [n_layers_req,
      P, H] (residual after each requested layer, post layer-scalar), mlp [.., P, d_ff] (GeGLU activations entering
      down_proj, float16), attn [.., heads, P, L] (attention probabilities, float16), logprobs [P] of the
      continuation tokens (and the closing delimiter) from the native cached forward, logprobs_manual [P] from the
      capture forward, top5_ids / top5_logprobs.
Edit arms apply the run's current edit (edit/*.npz, W' = W + B @ A) for that request only.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..edits import EditSet
from ..replay.codec import Q, split_prefix
from ..replay.engine import frozen_gen, fullcall_gen, seeds_for
from ..replay.records import FileStore, iter_calls
from ..scoring import EditScorer, ExactReplacer, SwegemmaReplacer

LIMITS = {"gen_calls": 128, "gen_draws": 2, "read_texts": 32, "read_layers": 8, "read_positions": 256,
          "attn_layers": 2, "continuation_tokens": 512}
HARNESS_VERSION = "weightprog-1"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class ServerConfig:
    runs_dir: str
    calls: str                       # calls.jsonl of the callable set
    files: str                       # files/<sha>.txt
    cache_dir: str
    model_dir: str
    calls_raw: str | None = None     # optional calls with the raw rendering (same keys; only `prompt` differs)
    backend: str = "mlx"
    dtype: str = "bfloat16"
    device: str | None = None
    replacer: str = "exact"          # exact | swegemma
    seed_policy: str = "key"         # per-(call, draw) seeds seed_of(key, draw); the only policy a server should use
    limits: dict = field(default_factory=lambda: dict(LIMITS))
    poll_seconds: float = 2.0

    @classmethod
    def load(cls, path: str | Path) -> "ServerConfig":
        d = json.loads(Path(path).read_text())
        base = Path(path).resolve().parent
        for k in ("runs_dir", "calls", "files", "cache_dir", "model_dir", "calls_raw"):
            if d.get(k) and not Path(d[k]).is_absolute():
                d[k] = str(base / d[k])
        return cls(**d)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def init_run(cfg: ServerConfig, run_id: str, proposer: str, model: str, effort: str, budget: int = 10,
             fullcall: bool = False) -> Path:
    run = Path(cfg.runs_dir) / run_id
    if run.exists():
        raise FileExistsError(run)
    for d in ("requests", "edit", "work"):
        (run / d).mkdir(parents=True)
    state = {"run_id": run_id, "proposer": proposer, "model": model, "effort": effort, "budget": int(budget), "used": 0,
             "created_utc": now(), "limits": cfg.limits, "harness_version": HARNESS_VERSION, "fullcall": bool(fullcall),
             "calls_sha256": _sha(Path(cfg.calls)), "backend": cfg.backend, "dtype": cfg.dtype}
    (run / "state.json").write_text(json.dumps(state, indent=1))
    (run / "notebook.md").write_text(f"# Notebook of run {run_id}\n\n")
    return run


def _strata_key(x: dict) -> str:
    return str(x["recorded_kind"]) + ("/" + str(x["recorded_strategy"]) if x["recorded_kind"] == "ok" else "")


def _paired(b: dict, e: dict, field_: str, labels=("both", "only_base", "only_edit", "neither")) -> dict:
    t = {k: 0 for k in labels}
    for k, x in b.items():
        y = e[k]
        t[labels[0] if x[field_] and y[field_] else labels[1] if x[field_] else labels[2] if y[field_] else labels[3]] += 1
    return t


class Server:
    def __init__(self, cfg: ServerConfig, backend=None):
        self.cfg = cfg
        if backend is None:
            from ..backends import get_backend
            backend = get_backend(cfg.backend, cfg.model_dir, dtype=cfg.dtype, device=cfg.device)
        self.be = backend
        self.q_id = self.be.tokenizer.token_to_id(Q)
        self.calls = {r["key"]: r for r in iter_calls(cfg.calls)}
        self.raw_prompts = {}
        if cfg.calls_raw:
            for r in iter_calls(cfg.calls_raw, only_usable=False):
                if r["key"] in self.calls:
                    self.raw_prompts[r["key"]] = r["prompt"]
            if set(self.raw_prompts) != set(self.calls):
                raise ValueError("calls_raw does not cover the callable set")
        self.files = FileStore(cfg.files)
        replacer = SwegemmaReplacer() if cfg.replacer == "swegemma" else ExactReplacer()
        self.scorer = EditScorer(replacer, file_text=self.files)
        self.cache_dir = Path(cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{self.be.name}"
        self._cache_files = {(m, r): self.cache_dir / f"base_{m}_{r}_{tag}.jsonl" for m in ("gen", "fullcall") for r in ("served", "raw")}
        self.base_cache = {k: {} for k in self._cache_files}
        for k, p in self._cache_files.items():
            if p.exists():
                for line in p.open():
                    g = json.loads(line)
                    self.base_cache[k][(g["key"], g["draw"])] = g
        self.L = cfg.limits

    # ---- helpers -----------------------------------------------------------------------------------------
    def call(self, key: str, rendering: str) -> dict:
        if rendering not in ("served", "raw"):
            raise ValueError("rendering must be served or raw")
        if rendering == "raw" and not self.raw_prompts:
            raise ValueError("this server has no raw rendering")
        r = self.calls[key]
        return dict(r, prompt=self.raw_prompts[key]) if rendering == "raw" else r

    def _edits(self, run_dir: Path) -> EditSet:
        d = run_dir / "edit"
        return EditSet.load_dir(d) if d.is_dir() else EditSet()

    def _generate(self, mode: str, keys, draws, arm, run_dir, rendering) -> list[dict]:
        fn = frozen_gen if mode == "gen" else fullcall_gen
        cache = self.base_cache[(mode, rendering)]
        gens = []
        es = self._edits(run_dir) if arm == "edit" else EditSet()
        if arm == "edit" and not es:
            raise ValueError("arm 'edit' but the run has no saved edit")
        with self.be.edits(es):
            for k in keys:
                if arm == "base" and all((k, d) in cache for d in range(draws)):
                    gens += [cache[(k, d)] for d in range(draws)]
                    continue
                g = fn(self.be, self.call(k, rendering), draws, seeds_for(self.cfg.seed_policy, k, 0, draws))
                gens += g
                if arm == "base":
                    with self._cache_files[(mode, rendering)].open("a") as f:
                        for x in g:
                            f.write(json.dumps(x) + "\n")
                            cache[(x["key"], x["draw"])] = x
        return gens

    # ---- gen ---------------------------------------------------------------------------------------------
    def _select(self, req: dict) -> list[str]:
        sel = req.get("calls")
        if isinstance(sel, dict):
            keys = sorted(self.calls)
            random.Random(int(sel.get("seed", 0))).shuffle(keys)
            keys = keys[: int(sel["sample"])]
        else:
            keys = list(sel or [])
        if not keys or len(keys) > self.L["gen_calls"]:
            raise ValueError(f"1..{self.L['gen_calls']} calls per pass")
        bad = [k for k in keys if k not in self.calls]
        if bad:
            raise ValueError(f"not callable-set calls: {bad[:3]}")
        return keys

    def do_gen(self, req: dict, run_dir: Path, work: Path, fullcall: bool = False) -> dict:
        arms = req.get("arms", ["base", "edit"])
        draws = int(req.get("draws", 1))
        if not set(arms) <= {"base", "edit"} or not 1 <= draws <= self.L["gen_draws"]:
            raise ValueError("arms must be base/edit; draws 1..%d" % self.L["gen_draws"])
        keys = self._select(req)
        rendering = req.get("rendering", "served")
        self.call(keys[0], rendering)
        result = {"n_calls": len(keys), "draws": draws, "rendering": rendering, "arms": {}}
        scored = {}
        for arm in arms:
            gens = self._generate("gen", keys, draws, arm, run_dir, rendering)
            rows = []
            for g in gens:
                x = self.scorer.score(self.calls[g["key"]], g["gen"], closed=g["closed"])
                x["draw"] = g["draw"]
                rows.append(x)
            scored[arm] = {(x["key"], x["draw"]): x for x in rows}
            n = len(rows)
            acc = [x for x in rows if x["accepted"]]
            strata = {}
            for x in rows:
                s = strata.setdefault(_strata_key(x), {"n": 0, "accepted": 0, "not_found": 0, "escape_artefact": 0})
                s["n"] += 1
                s["accepted"] += x["accepted"]
                s["not_found"] += int(x["outcome"] == "not_found")
                s["escape_artefact"] += x["escape_artefact"]
            result["arms"][arm] = {
                "n": n, "accepted": len(acc) / n, "not_found": sum(x["outcome"] == "not_found" for x in rows) / n,
                "multiple": sum(x["outcome"] == "multiple" for x in rows) / n, "exact_copy": sum(x["exact_copy"] for x in rows) / n,
                "escape_artefact": sum(x["escape_artefact"] for x in rows) / n, "unclosed": sum(1 - x["closed"] for x in rows) / n,
                "compile_fail_among_accepted": (sum(1 for x in acc if x["compiles"] == 0) / len(acc)) if acc else None,
                "by_recorded_stratum": strata}
            with (work / "rows.jsonl").open("a") as f:
                for x, g in zip(rows, gens):
                    f.write(json.dumps({"arm": arm, **x, "gen": g["gen"]}) + "\n")
        if set(arms) == {"base", "edit"}:
            result["paired_accepted"] = _paired(scored["base"], scored["edit"], "accepted",
                                                ("both_accepted", "only_base", "only_edit", "neither"))
            result["paired_escape_artefact"] = _paired(scored["base"], scored["edit"], "escape_artefact")
        if fullcall and req.get("fullcall", True):
            result["fullcall"] = self.do_fullcall(keys, draws, arms, rendering, run_dir, work)
        return result

    def do_fullcall(self, keys, draws, arms, rendering, run_dir: Path, work: Path) -> dict:
        usable, skipped = [], []
        for k in keys:
            try:
                split_prefix(self.calls[k])
                usable.append(k)
            except ValueError:
                skipped.append(k)
        out = {"n_calls": len(usable), "skipped": skipped, "arms": {}}
        if not usable:
            return out
        scored = {}
        for arm in arms:
            gens = self._generate("fullcall", usable, draws, arm, run_dir, rendering)
            rows = []
            for g in gens:
                x = self.scorer.score_fullcall(self.calls[g["key"]], g["gen"], g["gen_new"], g["closed"], g["closed_new"])
                x["draw"] = g["draw"]
                rows.append(x)
            scored[arm] = {(x["key"], x["draw"]): x for x in rows}
            n = len(rows)
            acc = [x for x in rows if x["accepted"]]
            strata = {}
            for x in rows:
                s = strata.setdefault(_strata_key(x), {"n": 0, "accepted": 0, "not_found": 0, "escape_artefact": 0, "compile_fail": 0})
                s["n"] += 1
                s["accepted"] += x["accepted"]
                s["not_found"] += int(x["outcome"] == "not_found")
                s["escape_artefact"] += x["artefact_old"]
                s["compile_fail"] += int(bool(x["accepted"]) and x["compiles"] == 0)
            out["arms"][arm] = {
                "n": n, "accepted": len(acc) / n, "not_found": sum(x["outcome"] == "not_found" for x in rows) / n,
                "exact_copy": sum(x["exact_copy"] for x in rows) / n, "escape_artefact": sum(x["artefact_old"] for x in rows) / n,
                "escape_artefact_new": sum(x["artefact_new"] for x in rows) / n, "unclosed_new": sum(1 - x["closed_new"] for x in rows) / n,
                "compile_fail_among_accepted": (sum(1 for x in acc if x["compiles"] == 0) / len(acc)) if acc else None,
                "by_recorded_stratum": strata}
            with (work / "rows_fullcall.jsonl").open("a") as f:
                for x, g in zip(rows, gens):
                    f.write(json.dumps({"arm": arm, **x, "gen_new": g["gen_new"], "gen": g["gen"]}) + "\n")
        if set(arms) == {"base", "edit"}:
            out["paired_accepted"] = _paired(scored["base"], scored["edit"], "accepted",
                                             ("both_accepted", "only_base", "only_edit", "neither"))
            out["paired_escape_artefact"] = _paired(scored["base"], scored["edit"], "artefact_old")
        return out

    # ---- read --------------------------------------------------------------------------------------------
    def do_read(self, req: dict, run_dir: Path, work: Path) -> dict:
        texts = req.get("texts", [])
        layers = sorted({int(l) for l in req.get("layers", [])})
        capture = set(req.get("capture", ["resid", "logprobs"]))
        if not capture <= {"resid", "mlp", "attn", "logprobs"}:
            raise ValueError("capture kinds: resid, mlp, attn, logprobs")
        attn_layers = sorted({int(l) for l in req.get("attn_layers", [])}) if "attn" in capture else []
        if not 1 <= len(texts) <= self.L["read_texts"] or len(layers) > self.L["read_layers"] or len(attn_layers) > self.L["attn_layers"]:
            raise ValueError(f"limits: texts 1..{self.L['read_texts']}, layers <= {self.L['read_layers']}, "
                             f"attn_layers <= {self.L['attn_layers']}")
        arm = req.get("arm", "base")
        rendering = req.get("rendering", "served")
        es = self._edits(run_dir) if arm == "edit" else EditSet()
        if arm == "edit" and not es:
            raise ValueError("arm 'edit' but the run has no saved edit")
        tok = self.be.tokenizer
        summaries = []
        with self.be.edits(es):
            for ti, t in enumerate(texts):
                if t.get("key") not in self.calls:
                    raise ValueError(f"text {ti}: key must be a callable-set call")
                r = self.call(t["key"], rendering)
                cont = r["old_string"] if t.get("continuation", "recorded") == "recorded" else str(t["continuation"])
                p_ids = tok.encode(r["prompt"] + r["prefix"])
                c_ids = tok.encode(cont)[: self.L["continuation_tokens"]]
                ids = p_ids + c_ids
                if req.get("positions", "continuation") == "prompt_last":
                    positions = [len(p_ids) - 1]
                else:
                    positions = list(range(len(p_ids) - 1, len(ids)))[: self.L["read_positions"]]
                what = ({"resid"} if "resid" in capture else set()) | ({"mlp"} if "mlp" in capture else set()) | \
                       ({"attn"} if "attn" in capture else set()) | ({"logits"} if "logprobs" in capture else set())
                cap = self.be.capture(ids, positions, layers=layers, what=what, attn_layers=attn_layers) if what else {}
                arrays = {"positions": np.array(positions), "token_ids": np.array(ids[min(positions):max(positions) + 2])}
                if "resid" in cap:
                    arrays["resid"] = cap["resid"].astype(np.float32)
                if "mlp" in cap:
                    arrays["mlp"] = cap["mlp"].astype(np.float16)
                if "attn" in cap:
                    arrays["attn"] = cap["attn"].astype(np.float16)
                summary = {"key": r["key"], "recorded_kind": r.get("error_kind"), "prompt_tokens": len(p_ids),
                           "continuation_tokens": len(c_ids), "n_positions": len(positions)}
                if "logprobs" in capture:
                    logp = self.be.logprobs(ids, positions)
                    lg = cap["logits"].astype(np.float64)
                    manual = lg - lg.max(-1, keepdims=True)
                    manual = manual - np.log(np.exp(manual).sum(-1, keepdims=True))
                    targets = (c_ids + [self.q_id])[: len(positions)]
                    ar = np.arange(len(targets))
                    lp = logp[ar, targets]
                    top = np.argsort(-logp, axis=-1)[:, :5]
                    arrays.update(logprobs=lp.astype(np.float32), logprobs_manual=manual[ar, targets].astype(np.float32),
                                  top5_ids=top, top5_logprobs=np.take_along_axis(logp, top, axis=-1).astype(np.float32))
                    summary.update(sum_logprob=float(lp.sum()), min_logprob=float(lp.min()),
                                   **{"first_below_-2": int(np.argmax(lp < -2)) if (lp < -2).any() else None})
                np.savez_compressed(work / f"text_{ti}.npz", **arrays)
                summaries.append(summary)
        return {"arm": arm, "rendering": rendering, "layers": layers, "attn_layers": attn_layers,
                "capture": sorted(capture), "texts": summaries, "files": f"work/{work.name}/text_<i>.npz"}

    # ---- queue -------------------------------------------------------------------------------------------
    def handle(self, req_path: Path) -> dict:
        run_dir = req_path.parent.parent
        state_p = run_dir / "state.json"
        claimed = req_path.with_suffix(".json.claimed")
        req_path.rename(claimed)
        done = req_path.with_name(req_path.stem + ".done.json")
        state = json.loads(state_p.read_text())
        t0 = time.time()
        rec = {"request": req_path.name, "received_utc": now()}
        try:
            req = json.loads(claimed.read_text())
            snap = EditSet.snapshot(run_dir / "edit") if (run_dir / "edit").is_dir() else {}
            if req.get("expect_edit") is not None and req["expect_edit"] != snap:
                raise ValueError("expect_edit does not match the run's current edit/ (pinned snapshot mismatch)")
            rec["edit_snapshot"] = snap
            if state["used"] >= state["budget"]:
                rec.update(status="refused", reason=f"budget exhausted ({state['used']}/{state['budget']})")
            else:
                work = run_dir / "work" / req_path.stem
                work.mkdir(parents=True, exist_ok=True)
                if req.get("type") == "gen":
                    res = self.do_gen(req, run_dir, work, fullcall=bool(state.get("fullcall")))
                elif req.get("type") == "read":
                    res = self.do_read(req, run_dir, work)
                else:
                    raise ValueError("type must be gen or read")
                state["used"] += 1
                state.setdefault("passes", []).append({"request": req_path.name, "type": req["type"], "utc": now(),
                                                       "secs": round(time.time() - t0, 1), "start_utc": rec["received_utc"],
                                                       "edit_snapshot": snap})
                tmp = state_p.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(state, indent=1))
                tmp.replace(state_p)
                rec.update(status="ok", result=res)
        except Exception as e:  # noqa: BLE001
            rec.update(status="failed (not counted)", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
        rec.update(secs=round(time.time() - t0, 1), passes_used=state["used"], budget=state["budget"])
        done.write_text(json.dumps(rec, indent=1))
        print(f"[{now()}] {run_dir.name}/{req_path.name}: {rec['status']} ({rec['secs']}s, {state['used']}/{state['budget']})", flush=True)
        return rec

    def pending(self) -> list[Path]:
        ps = [p for p in Path(self.cfg.runs_dir).glob("*/requests/fwd_*.json") if not p.name.endswith(".done.json")]

        def order(p: Path):
            try:
                k = int(p.stem.split("_")[1])
            except (IndexError, ValueError):
                k = 10 ** 9
            return (p.parent.parent.name, k)

        return sorted(ps, key=order)

    def poll_once(self) -> list[dict]:
        return [self.handle(p) for p in self.pending()]

    def loop(self, once: bool = False) -> None:
        while True:
            self.poll_once()
            if once:
                return
            time.sleep(self.cfg.poll_seconds)


def spec_text(limits: dict | None = None) -> str:
    """The server specification a proposer packet ships (module docstring + the frozen LIMITS)."""
    return ("# Forward-pass server specification\n\n```\n" + (__doc__ or "").strip() + "\n```\n\nLIMITS = "
            + json.dumps(limits or LIMITS) + "\n")


def config_dict(cfg: ServerConfig) -> dict:
    return asdict(cfg)
