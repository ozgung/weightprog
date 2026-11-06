"""Dependency-free safetensors reader and writer on numpy, with bfloat16 support.

numpy has no bfloat16 dtype, so `safetensors.numpy` cannot read the bf16 checkpoints Gemma 4 ships in. This module
memory-maps the file, parses the JSON header itself and widens BF16 to float32 exactly (a bf16 value is the upper 16
bits of the float32 with the same value). Reading is lazy: only the tensors you ask for are touched.

    r = SafetensorsReader("model.safetensors")       # or a directory with model*.safetensors / an index.json
    r.keys(); r.shape(name); r.dtype(name)
    W = r.get(name)                                  # float32 for F32/F16/BF16 (copy), native dtype otherwise

    save_file({"w": np.ones((2, 3), np.float32)}, "x.safetensors", dtypes={"w": "BF16"})
"""

from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path

import numpy as np

_NP = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32,
       "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_, "BF16": np.uint16}
_FROM_NP = {np.dtype(np.float64): "F64", np.dtype(np.float32): "F32", np.dtype(np.float16): "F16",
            np.dtype(np.int64): "I64", np.dtype(np.int32): "I32", np.dtype(np.int16): "I16",
            np.dtype(np.int8): "I8", np.dtype(np.uint8): "U8", np.dtype(np.bool_): "BOOL"}


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """uint16 bf16 bit patterns -> float32 (exact)."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> uint16 bf16 bit patterns, round to nearest even (NaN kept as a quiet NaN)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    rounding = ((u >> 16) & 1) + np.uint32(0x7FFF)
    out = ((u + rounding) >> 16).astype(np.uint16)
    nan = np.isnan(x)
    if nan.any():
        out[nan] = np.uint16(0x7FC0)
    return out


class _File:
    def __init__(self, path: Path):
        self.path = path
        self._fh = path.open("rb")
        n = struct.unpack("<Q", self._fh.read(8))[0]
        self.header = json.loads(self._fh.read(n))
        self.metadata = self.header.pop("__metadata__", None) or {}
        self.base = 8 + n
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def raw(self, name: str) -> np.ndarray:
        h = self.header[name]
        s, e = h["data_offsets"]
        arr = np.frombuffer(self._mm, dtype=_NP[h["dtype"]], count=(e - s) // np.dtype(_NP[h["dtype"]]).itemsize,
                            offset=self.base + s)
        return arr.reshape(h["shape"])

    def close(self) -> None:
        self._mm.close()
        self._fh.close()


class SafetensorsReader:
    """Lazy reader over one .safetensors file, a directory of shards, or a model.safetensors.index.json."""

    def __init__(self, path: str | Path):
        p = Path(path)
        if p.is_dir():
            idx = p / "model.safetensors.index.json"
            files = sorted({p / f for f in json.loads(idx.read_text())["weight_map"].values()}) if idx.exists() \
                else sorted(p.glob("model*.safetensors"))
        elif p.suffix == ".json":
            files = sorted({p.parent / f for f in json.loads(p.read_text())["weight_map"].values()})
        else:
            files = [p]
        if not files:
            raise FileNotFoundError(f"no safetensors under {path}")
        self._files = [_File(f) for f in files]
        self._where = {k: f for f in self._files for k in f.header}

    def keys(self) -> list[str]:
        return sorted(self._where)

    def __contains__(self, name: str) -> bool:
        return name in self._where

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._where[name].header[name]["shape"])

    def dtype(self, name: str) -> str:
        return self._where[name].header[name]["dtype"]

    def get(self, name: str, float32: bool = True) -> np.ndarray:
        """The tensor as a numpy array (a copy). Floating types are widened to float32 unless float32=False
        (then BF16 is returned as its uint16 bit pattern)."""
        if name not in self._where:
            raise KeyError(name)
        f = self._where[name]
        a = f.raw(name)
        dt = f.header[name]["dtype"]
        if dt == "BF16":
            return bf16_bits_to_f32(a) if float32 else a.copy()
        if float32 and dt in ("F16", "F64"):
            return a.astype(np.float32)
        return a.copy()

    def rows(self, name: str, idx) -> np.ndarray:
        """Selected rows of a 2-D tensor as float32, without widening the whole tensor (e.g. unembedding rows)."""
        f = self._where[name]
        a = f.raw(name)[np.asarray(idx)]
        return bf16_bits_to_f32(a) if f.header[name]["dtype"] == "BF16" else a.astype(np.float32)

    def close(self) -> None:
        for f in self._files:
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def save_file(tensors: dict[str, np.ndarray], path: str | Path, dtypes: dict[str, str] | None = None,
              default_dtype: str | None = None, metadata: dict[str, str] | None = None) -> None:
    """Write numpy arrays as safetensors. dtypes[name] (or default_dtype) may be "BF16", "F16" or "F32" to convert
    floating arrays; otherwise the array's own dtype is kept."""
    dtypes = dtypes or {}
    header: dict = {}
    blobs = []
    off = 0
    for name in sorted(tensors):
        a = np.asarray(tensors[name])
        want = dtypes.get(name, default_dtype if np.issubdtype(a.dtype, np.floating) else None)
        if want == "BF16":
            b = f32_to_bf16_bits(a.astype(np.float32))
            dt = "BF16"
        elif want in ("F16", "F32", "F64"):
            b = np.ascontiguousarray(a, dtype=_NP[want])
            dt = want
        else:
            b = np.ascontiguousarray(a)
            dt = _FROM_NP[b.dtype]
        raw = b.tobytes()
        header[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with Path(path).open("wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for raw in blobs:
            f.write(raw)
