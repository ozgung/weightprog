"""Low-rank module edits W' = W + B @ A: the unit every proposer submits and every gate judges.

    from weightprog.edits import EditSet, ModuleEdit
    es = EditSet.load_dir("run/edit")             # *.npz with keys module, B, A, meta (the original kernel's format)
    es.validate(spec, weights)                      # allowed module, shapes, rank <= 128, finite, no whole-tensor rescale
    es.norms(weights)                               # per module: rank, ||BA||_F, ||W||_F, relative change
    es.save_dir("out/edit"); es.snapshot()          # {file name: sha256}, as the server pins with expect_edit
    export_peft(es, "adapter_dir", spec)            # adapter_config.json + adapter_model.safetensors, delta == B @ A

`weights` is anything with `get(name) -> np.ndarray` and `shape(name)` (a SafetensorsReader, a backend's
read_weight wrapped by `WeightSource`, or a dict).
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .names import EDITABLE_RE, ModelSpec, canonical, file_stem

MAX_RANK = 128
RESCALE_TOL = 1e-4


class EditError(ValueError):
    pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class ModuleEdit:
    module: str
    B: np.ndarray  # [out, r]
    A: np.ndarray  # [r, in]
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.module = canonical(self.module)
        self.B = np.asarray(self.B, dtype=np.float32)
        self.A = np.asarray(self.A, dtype=np.float32)

    @property
    def rank(self) -> int:
        return int(self.B.shape[1]) if self.B.ndim == 2 else -1

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.B.shape[0]), int(self.A.shape[1])

    def delta(self) -> np.ndarray:
        """B @ A in float32 (what every backend adds to W in float32 before casting back)."""
        return self.B @ self.A

    def scaled(self, factor: float) -> "ModuleEdit":
        return ModuleEdit(self.module, self.B * np.float32(factor), self.A.copy(), dict(self.meta, scaled_by=factor))

    def file_name(self) -> str:
        return file_stem(self.module) + ".npz"

    def to_npz_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez(buf, module=np.array(self.module), B=self.B, A=self.A, meta=np.array(json.dumps(self.meta)))
        return buf.getvalue()

    @classmethod
    def from_npz(cls, path: str | Path) -> "ModuleEdit":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        return cls(str(z["module"]), z["B"], z["A"], meta)


def check_module(edit: ModuleEdit, spec: ModelSpec | None = None, weights=None, max_rank: int = MAX_RANK,
                 refuse_rescale: bool = True) -> dict:
    """Validate one edit as the original kernel's save_edit did; return its norm record (raises EditError)."""
    name = edit.module
    if not EDITABLE_RE.match(name):
        raise EditError(f"module not editable: {name}")
    if spec is not None and not spec.is_editable(name):
        raise EditError(f"module not in this model: {name}")
    if weights is not None and hasattr(weights, "__contains__") and name not in weights:
        raise EditError(f"module not in the checkpoint: {name}")
    if spec is not None:
        out, inn = spec.module_shape(name)
    elif weights is not None:
        out, inn = tuple(weights.shape(name))
    else:
        out, inn = edit.shape
    B, A = edit.B, edit.A
    if B.ndim != 2 or A.ndim != 2 or B.shape[0] != out or A.shape[1] != inn or B.shape[1] != A.shape[0]:
        raise EditError(f"shapes: need B [{out}, r] and A [r, {inn}], got {B.shape} and {A.shape}")
    r = B.shape[1]
    if not 1 <= r <= max_rank:
        raise EditError(f"rank {r} not in 1..{max_rank}")
    if not (np.isfinite(B).all() and np.isfinite(A).all()):
        raise EditError("non-finite values")
    D = B @ A
    rec = {"module": name, "rank": int(r), "delta_fro": float(np.linalg.norm(D))}
    if weights is not None:
        W = np.asarray(weights.get(name), dtype=np.float32)
        wn = float(np.linalg.norm(W))
        rec.update(weight_fro=wn, rel_change=float(np.linalg.norm(D) / wn) if wn else float("inf"))
        if refuse_rescale and wn > 0:
            c = float((W * D).sum() / (wn ** 2))
            if np.linalg.norm(D - c * W) <= RESCALE_TOL * max(1e-12, float(np.linalg.norm(D))):
                raise EditError("refused: the edit is a multiple of the whole tensor (rescale)")
    return rec


class EditSet:
    """An ordered set of module edits, at most one per module (adding again replaces)."""

    def __init__(self, edits: Iterable[ModuleEdit] = ()):
        self._e: dict[str, ModuleEdit] = {}
        for e in edits:
            self.add(e)

    # ---- container ---------------------------------------------------------------------------------------
    def add(self, e: ModuleEdit) -> None:
        self._e[e.module] = e

    def remove(self, module: str) -> bool:
        return self._e.pop(canonical(module), None) is not None

    def __iter__(self) -> Iterator[ModuleEdit]:
        return iter(sorted(self._e.values(), key=lambda e: e.file_name()))

    def __len__(self) -> int:
        return len(self._e)

    def __bool__(self) -> bool:
        return bool(self._e)

    def __getitem__(self, module: str) -> ModuleEdit:
        return self._e[canonical(module)]

    def __contains__(self, module: str) -> bool:
        return canonical(module) in self._e

    def modules(self) -> list[str]:
        return [e.module for e in self]

    def scaled(self, factor: float) -> "EditSet":
        return EditSet(e.scaled(factor) for e in self)

    # ---- io ----------------------------------------------------------------------------------------------
    @classmethod
    def load_dir(cls, d: str | Path) -> "EditSet":
        d = Path(d)
        if not d.is_dir():
            raise FileNotFoundError(d)
        es = cls()
        for f in sorted(d.glob("*.npz")):
            e = ModuleEdit.from_npz(f)
            if e.module in es:
                raise EditError(f"two files edit {e.module}")
            es.add(e)
        return es

    @classmethod
    def load(cls, path: str | Path) -> "EditSet":
        """A run folder (uses its edit/), an edit directory, or a single .npz file."""
        p = Path(path)
        if p.is_file():
            return cls([ModuleEdit.from_npz(p)])
        if (p / "edit").is_dir():
            return cls.load_dir(p / "edit")
        return cls.load_dir(p)

    def save_dir(self, d: str | Path, clear: bool = False) -> list[Path]:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        if clear:
            for f in d.glob("*.npz"):
                f.unlink()
        out = []
        for e in self:
            f = d / e.file_name()
            f.write_bytes(e.to_npz_bytes())
            out.append(f)
        return out

    @staticmethod
    def snapshot(d: str | Path) -> dict[str, str]:
        """{file name: sha256} of the *.npz files in d (the server's expect_edit pin)."""
        return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path(d).glob("*.npz"))}

    # ---- checks ------------------------------------------------------------------------------------------
    def validate(self, spec: ModelSpec | None = None, weights=None, max_rank: int = MAX_RANK,
                 refuse_rescale: bool = True) -> list[dict]:
        return [check_module(e, spec, weights, max_rank, refuse_rescale) for e in self]

    def norms(self, weights=None) -> list[dict]:
        out = []
        for e in self:
            D = e.delta()
            rec = {"module": e.module, "rank": e.rank, "delta_fro": float(np.linalg.norm(D))}
            if weights is not None:
                wn = float(np.linalg.norm(np.asarray(weights.get(e.module), dtype=np.float32)))
                rec.update(weight_fro=wn, rel_change=float(np.linalg.norm(D) / wn) if wn else float("inf"))
            out.append(rec)
        return out

    def summary(self) -> dict:
        return {"modules": self.modules(), "ranks": [e.rank for e in self],
                "delta_fro": [float(np.linalg.norm(e.delta())) for e in self]}


class WeightSource:
    """Adapter giving `get(name)` / `shape(name)` / `in` over a backend's read_weight or a dict of arrays."""

    def __init__(self, getter, shapes: dict | None = None, names: Iterable[str] | None = None):
        self._get = getter if callable(getter) else (lambda n: getter[n])
        self._names = set(names) if names is not None else (set(getter) if isinstance(getter, dict) else None)
        self._shapes = shapes or {}

    def get(self, name: str) -> np.ndarray:
        return np.asarray(self._get(canonical(name)), dtype=np.float32)

    def shape(self, name: str):
        n = canonical(name)
        return self._shapes.get(n) or tuple(self.get(n).shape)

    def __contains__(self, name: str) -> bool:
        return True if self._names is None else canonical(name) in self._names


# ---- PEFT export ---------------------------------------------------------------------------------------------

def export_peft(edits: EditSet, out_dir: str | Path, base_model: str = "", module_prefix: str = "model.language_model.",
                peft_prefix: str = "base_model.model.", dtype: str = "F32") -> Path:
    """Write a PEFT LoRA adapter whose delta equals B @ A exactly.

    PEFT computes delta = lora_B @ lora_A * (lora_alpha / r). We store lora_A = A, lora_B = B and set
    lora_alpha = r per module (rank_pattern / alpha_pattern when ranks differ), so the scaling is exactly 1.
    Module paths are those of the Hugging Face Gemma4ForConditionalGeneration (`model.language_model.layers...`);
    pass module_prefix="model." for a text-only Gemma4ForCausalLM checkpoint. Tensors are written in float32 by
    default (dtype="BF16" halves the size but rounds B and A, so the delta is no longer exact).
    """
    from .io.safetensors_np import save_file

    if not edits:
        raise EditError("empty edit set")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensors, targets, ranks = {}, [], {}
    for e in edits:
        path = module_prefix + e.module.removeprefix("model.language_model.").removesuffix(".weight")
        targets.append(path)
        ranks[path] = e.rank
        tensors[f"{peft_prefix}{path}.lora_A.weight"] = e.A
        tensors[f"{peft_prefix}{path}.lora_B.weight"] = e.B
    r0 = max(ranks.values())
    rank_pattern = {re.escape(p): r for p, r in ranks.items() if r != r0}
    cfg = {
        "peft_type": "LORA", "task_type": None, "base_model_name_or_path": base_model, "r": r0, "lora_alpha": r0,
        "rank_pattern": rank_pattern, "alpha_pattern": dict(rank_pattern), "target_modules": sorted(targets),
        "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False, "use_rslora": False, "use_dora": False,
        "init_lora_weights": True, "inference_mode": True, "modules_to_save": None, "layers_to_transform": None,
        "layers_pattern": None, "revision": None,
    }
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    (out / "weightprog_export.json").write_text(json.dumps({
        "note": "delta = lora_B @ lora_A exactly (lora_alpha == r per module)", "exported_utc": _now(),
        "modules": [e.module for e in edits], "norms": edits.norms()}, indent=1))
    save_file(tensors, out / "adapter_model.safetensors", default_dtype=dtype, metadata={"format": "pt"})
    return out
