"""The one interface every backend implements, plus the pieces they share (tokenizer, sampling, decode segments).

A backend owns one loaded Gemma 4 model and offers exactly the operations a proposer, a replay or a gate needs:

    be = get_backend("mlx")            # or "torch"
    be.load(model_dir, dtype="bfloat16")
    ids = be.tokenizer.encode(text)
    with be.edits(edit_set):           # W' = W + B @ A in float32, cast back; restored exactly on exit
        outs = be.generate(ids, max_new=64, draws=2, seeds=[s0, s1], stop_ids=[q_id])
        segs = be.decode_segments(ids, [Generate(256, [q_id]), Force(sep_ids), Generate(128, [q_id])], draws=2, seeds=...)
    lp = be.logprobs(ids, positions)   # [P, V] log-softmax from the native cached forward
    cap = be.capture(ids, positions, layers={40, 41}, what={"resid", "mlp", "attn", "logits"}, attn_layers={41})
    W = be.read_weight("model.language_model.layers.41.mlp.down_proj.weight")   # float32 numpy

Conventions shared by all backends (so runs are comparable within a backend):
- generate / decode_segments prefill ids[:-1] once (in chunks of `prefill_chunk` tokens), copy the cache per draw and
  feed ids[-1] first. Draw 0 is greedy; draws >= 1 sample with `Sampling` (default T 0.6, top_p 0.95, top_k 64:
  the served settings). The sampler follows mlx_lm's order: top-p on the untempered distribution, then top-k, then
  categorical at temperature T. Each draw re-seeds the backend's RNG with seeds[d] once; the RNG then runs through
  every segment of that draw.
- A stop token ends a Generate segment and is NOT fed to the cache; a following Force segment usually starts with it.
- Positions: `logprobs(ids, positions)[j]` is the distribution predicting token positions[j] + 1. `capture` returns
  arrays with the positions on the second-to-last axis (resid [n_layers_req, P, H], mlp [.., P, d_ff],
  attn [.., heads, P, L], logits [P, V]).
- Sampled draws are NOT token-identical across backends (different RNGs); greedy agrees up to numerics. Never mix
  generations from different backends in one paired analysis.
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol, Sequence, runtime_checkable

import numpy as np

from ..names import ModelSpec

Q = '<|"|>'  # Gemma 4's string delimiter inside tool calls
CAPTURE_KINDS = ("resid", "resid_pre", "resid_mid", "mlp", "mlp_out", "attn", "pli", "logits")


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 64


SERVED = Sampling()


@dataclass(frozen=True)
class Generate:
    """Sample until a stop token (not fed) or max_new tokens."""
    max_new: int
    stop_ids: tuple[int, ...] = ()

    def __init__(self, max_new: int, stop_ids: Iterable[int] = ()):
        object.__setattr__(self, "max_new", int(max_new))
        object.__setattr__(self, "stop_ids", tuple(int(t) for t in stop_ids))


@dataclass(frozen=True)
class Force:
    """Teacher-force these tokens into the cache (their predictions are not sampled)."""
    ids: tuple[int, ...]

    def __init__(self, ids: Iterable[int]):
        object.__setattr__(self, "ids", tuple(int(t) for t in ids))


@dataclass
class SegmentResult:
    tokens: list[int] = field(default_factory=list)
    closed: int = 0          # 1 if a stop token ended the segment (Generate only)
    stop_id: int | None = None
    skipped: bool = False    # not run because an earlier Generate segment ended unclosed


class Tokenizer:
    """Thin wrapper over `tokenizers.Tokenizer` (no special tokens added; specials decoded verbatim)."""

    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer as _T

        p = Path(path)
        self._t = _T.from_file(str(p / "tokenizer.json" if p.is_dir() else p))

    def encode(self, text: str) -> list[int]:
        return self._t.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._t.decode([int(i) for i in ids], skip_special_tokens=False)

    def token_to_id(self, tok: str) -> int | None:
        return self._t.token_to_id(tok)

    def id_to_token(self, i: int) -> str | None:
        return self._t.id_to_token(int(i))

    @property
    def vocab_size(self) -> int:
        return self._t.get_vocab_size()

    def vocab(self) -> dict[str, int]:
        return self._t.get_vocab()


def seed_of(key: str, draw: int) -> int:
    """Per-(call, draw) seed of the forward-pass server: first 8 hex digits of sha1("key|draw")."""
    return int(hashlib.sha1(f"{key}|{draw}".encode()).hexdigest()[:8], 16)


def index_seed(index: int, draw: int) -> int:
    """Seed of the stand-alone replay scripts: 7919 * call index + draw."""
    return 7919 * index + draw


def plan_segments(segments: Sequence[Generate | Force]) -> None:
    if not segments:
        raise ValueError("no segments")
    for s in segments:
        if not isinstance(s, (Generate, Force)):
            raise TypeError(f"segment {s!r} is neither Generate nor Force")


def check_capture_request(what: Iterable[str]) -> set[str]:
    w = set(what)
    bad = w - set(CAPTURE_KINDS)
    if bad:
        raise ValueError(f"unknown capture kinds {sorted(bad)}; choose from {CAPTURE_KINDS}")
    return w


@runtime_checkable
class Backend(Protocol):
    name: str
    spec: ModelSpec
    tokenizer: Tokenizer
    prefill_chunk: int

    def load(self, model_dir: str | Path, dtype: str = "bfloat16", device: str | None = None) -> "Backend": ...

    def edits(self, edit_set) -> contextlib.AbstractContextManager: ...

    def generate(self, ids: Sequence[int], max_new: int, draws: int = 1, seeds: Sequence[int] | None = None,
                 stop_ids: Iterable[int] = (), sampling: Sampling = SERVED) -> list[SegmentResult]: ...

    def decode_segments(self, ids: Sequence[int], segments: Sequence[Generate | Force], draws: int = 1,
                        seeds: Sequence[int] | None = None, sampling: Sampling = SERVED,
                        abort_on_unclosed: bool = True) -> list[list[SegmentResult]]: ...

    def logprobs(self, ids: Sequence[int], positions: Sequence[int]) -> np.ndarray: ...

    def capture(self, ids: Sequence[int], positions: Sequence[int], layers: Iterable[int] = (),
                what: Iterable[str] = ("resid",), attn_layers: Iterable[int] = ()) -> dict[str, np.ndarray]: ...

    def read_weight(self, name: str) -> np.ndarray: ...

    def weight_names(self) -> list[str]: ...


class BackendBase:
    """Shared plumbing: edit bookkeeping and the generate-in-terms-of-segments wrapper."""

    name = "base"
    prefill_chunk = 2048

    def generate(self, ids, max_new, draws=1, seeds=None, stop_ids=(), sampling=SERVED):
        res = self.decode_segments(ids, [Generate(max_new, stop_ids)], draws=draws, seeds=seeds, sampling=sampling)
        return [r[0] for r in res]

    @contextlib.contextmanager
    def edits(self, edit_set):
        """Apply W' = W + B @ A (float32 add, cast to W's dtype) for the block; restore the exact originals."""
        originals = self._apply(edit_set) if edit_set else {}
        try:
            yield originals
        finally:
            if originals:
                self._restore(originals)

    def _seeds(self, draws: int, seeds):
        if seeds is None:
            seeds = list(range(draws))
        if len(seeds) < draws:
            raise ValueError(f"{draws} draws need {draws} seeds")
        return [int(s) for s in seeds[:draws]]

    @staticmethod
    def _positions(positions) -> list[int]:
        pos = [int(p) for p in positions]
        if not pos:
            raise ValueError("no positions")
        return pos

    def edited_modules(self, edit_set) -> list[str]:
        return [e.module for e in edit_set]
