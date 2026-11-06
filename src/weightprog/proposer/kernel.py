"""Proposer-side weight kernel: read a Gemma 4 checkpoint's tensors and save LoRA-expressible edits. No model runs here.

    import kernel as K                      # in a proposer packet (this file + safetensors_np.py + kernel_config.json)
    # or: from weightprog.proposer import kernel as K; K.configure(model_dir)
    K.names(r"layers\\.41\\..*mlp")          # tensor names matching a regex
    W = K.load("model.language_model.layers.41.mlp.down_proj.weight")    # float32 numpy, [out, in]
    E = K.rows("model.language_model.embed_tokens.weight", [ids...])     # selected rows, float32
    K.save_edit(RUN_DIR, name, B, A, note="why")     # W' = W + B @ A; B [out, r], A [r, in], r <= 128
    K.list_edits(RUN_DIR); K.remove_edit(RUN_DIR, name)

An edit on a module is W' = W + B @ A (one file per module in RUN_DIR/edit/; saving again replaces it). Allowed
modules: q/k/v/o/gate/up/down projections of the decoder layers that exist in the checkpoint (KV-shared layers have
no k/v projections; E4B: 258 modules). Refused: embeddings, per-layer embeddings and projections, norms, layer
scalars, any non-finite value, rank > 128, and any B @ A that is a multiple of W (a whole-tensor rescale).
This file depends only on numpy and safetensors_np.py, so it can be copied into a sandboxed packet.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np

try:
    from ..io.safetensors_np import SafetensorsReader
except ImportError:  # packet copy: safetensors_np.py sits next to this file
    from safetensors_np import SafetensorsReader  # type: ignore

ALLOWED = re.compile(r"^model\.language_model\.layers\.(\d+)\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$")
MAX_RANK = 128
RESCALE_TOL = 1e-4


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def fname(name: str) -> str:
    return name.removeprefix("model.language_model.").removesuffix(".weight").replace(".", "_") + ".npz"


class Kernel:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.reader = SafetensorsReader(self.model_dir)
        cfg_p = self.model_dir / "config.json"
        self.first_shared = None
        if cfg_p.exists():
            cfg = json.loads(cfg_p.read_text())
            t = cfg.get("text_config") or cfg
            self.first_shared = t["num_hidden_layers"] - int(t.get("num_kv_shared_layers") or 0)

    def names(self, pattern: str = ".*") -> list[str]:
        rx = re.compile(pattern)
        return sorted(k for k in self.reader.keys() if rx.search(k))

    def shape(self, name: str) -> tuple[int, ...]:
        return self.reader.shape(name)

    def load(self, name: str) -> np.ndarray:
        return self.reader.get(name)

    def rows(self, name: str, idx) -> np.ndarray:
        return self.reader.rows(name, idx)

    def allowed(self, name: str) -> bool:
        m = ALLOWED.match(name)
        if not m or name not in self.reader:
            return False
        layer, proj = int(m.group(1)), m.group(2)
        if self.first_shared is not None and layer >= self.first_shared and proj in ("self_attn.k_proj", "self_attn.v_proj"):
            return False
        return True

    def save_edit(self, run_dir: str | Path, name: str, B, A, note: str = "") -> dict:
        if not self.allowed(name):
            raise ValueError(f"module not editable: {name}")
        out, inn = self.shape(name)
        B = np.asarray(B, dtype=np.float32)
        A = np.asarray(A, dtype=np.float32)
        if B.ndim != 2 or A.ndim != 2 or B.shape[0] != out or A.shape[1] != inn or B.shape[1] != A.shape[0]:
            raise ValueError(f"shapes: need B [{out}, r] and A [r, {inn}], got {B.shape} and {A.shape}")
        r = B.shape[1]
        if not 1 <= r <= MAX_RANK:
            raise ValueError(f"rank {r} not in 1..{MAX_RANK}")
        if not (np.isfinite(B).all() and np.isfinite(A).all()):
            raise ValueError("non-finite values")
        W = self.load(name)
        D = B @ A
        wn = float(np.linalg.norm(W))
        c = float((W * D).sum() / (wn ** 2))
        if np.linalg.norm(D - c * W) <= RESCALE_TOL * max(1e-12, float(np.linalg.norm(D))):
            raise ValueError("refused: the edit is a multiple of the whole tensor (rescale)")
        meta = {"module": name, "rank": int(r), "delta_fro": float(np.linalg.norm(D)), "weight_fro": wn,
                "rel_change": float(np.linalg.norm(D) / wn), "note": note, "saved_utc": _now()}
        d = Path(run_dir) / "edit"
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / fname(name), module=np.array(name), B=B, A=A, meta=np.array(json.dumps(meta)))
        _log(run_dir, {"op": "save_edit", **meta})
        return meta

    def remove_edit(self, run_dir: str | Path, name: str) -> bool:
        f = Path(run_dir) / "edit" / fname(name)
        if f.exists():
            f.unlink()
            _log(run_dir, {"op": "remove_edit", "module": name, "removed_utc": _now()})
            return True
        return False

    @staticmethod
    def list_edits(run_dir: str | Path) -> list[dict]:
        out = []
        for f in sorted((Path(run_dir) / "edit").glob("*.npz")):
            z = np.load(f, allow_pickle=False)
            out.append(json.loads(str(z["meta"])))
        return out


def _log(run_dir, rec: dict) -> None:
    with (Path(run_dir) / "kernel_log.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


_K: Kernel | None = None


def configure(model_dir: str | Path) -> Kernel:
    global _K
    _K = Kernel(model_dir)
    return _K


def _k() -> Kernel:
    global _K
    if _K is None:
        md = os.environ.get("WEIGHTPROG_MODEL_DIR")
        cfg = Path(__file__).with_name("kernel_config.json")
        if not md and cfg.exists():
            md = json.loads(cfg.read_text())["model_dir"]
        if not md:
            raise RuntimeError("no model directory: call configure(dir), set WEIGHTPROG_MODEL_DIR or ship kernel_config.json")
        _K = Kernel(md)
    return _K


def names(pattern: str = ".*") -> list[str]:
    return _k().names(pattern)


def shape(name: str) -> tuple[int, ...]:
    return _k().shape(name)


def load(name: str) -> np.ndarray:
    return _k().load(name)


def rows(name: str, idx) -> np.ndarray:
    return _k().rows(name, idx)


def allowed(name: str) -> bool:
    return _k().allowed(name)


def save_edit(run_dir, name, B, A, note: str = "") -> dict:
    return _k().save_edit(run_dir, name, B, A, note)


def remove_edit(run_dir, name) -> bool:
    return _k().remove_edit(run_dir, name)


def list_edits(run_dir) -> list[dict]:
    return Kernel.list_edits(run_dir)
