"""Describe the behaviour you want to change, and the behaviour that must not change, as a list of cases.

A case is one prompt plus a rule that judges what the model does with it. Cases live in a JSONL file, one object per
line:

    {"id": "dump-01", "messages": [{"role": "user", "content": "..."}], "prefill": "```python\\n",
     "role": "target", "split": "tune", "group": "fastapi-style",
     "good": ["\\\\.model_dump\\\\("], "bad": ["\\\\.dict\\\\("], "max_new": 96}

Fields (all optional except `id` and one of `prompt` / `messages`):

| field | meaning |
|---|---|
| `prompt` | raw text fed to the model (already in its chat format if it is a chat model) |
| `messages`, `prefill` | chat turns rendered with the checkpoint's template (`weightprog.prompts.render_chat`), then `prefill` |
| `role` | `target`: the behaviour to change. `keep`: behaviour that must survive the edit (hard negatives, look-alikes, unrelated skills) |
| `split` | any label; by convention `tune` (yours to look at) and `holdout` (kept away until the end) |
| `group` | a cluster label (source, template family, repository). Statistics resample groups, and holdout should not share groups with tune |
| `good`, `bad` | regular expressions searched in the generated text. A generation is `bad` if any `bad` pattern matches, else `good` if any `good` pattern matches, else `other` |
| `check` | `"module:function"`; called as `function(text, case)` and returning `"good"`, `"bad"` or `"other"` (or True/False). Overrides `good`/`bad` |
| `contrast` | `{"good": "...", "bad": "..."}`: no generation; the case is scored by log P(good) - log P(bad) as continuations of the prompt |
| `max_new`, `stop` | generation length cap, and strings that end the generation (a stop string that is one vocabulary token, such as a code fence, ends decoding at once; longer ones are cut after the fact) |
| `meta` | anything else you want to keep with the case |

A `keep` case without `good`/`bad`/`check`/`contrast` is judged by "unchanged": the edited model's greedy text must
equal the base model's.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

ROLES = ("target", "keep")
LABELS = ("good", "bad", "other")


@dataclass
class Case:
    id: str
    prompt: str | None = None
    messages: list[dict] | None = None
    prefill: str = ""
    role: str = "target"
    split: str = "tune"
    group: str = ""
    good: list[str] = field(default_factory=list)
    bad: list[str] = field(default_factory=list)
    check: str | None = None
    contrast: dict | None = None
    max_new: int = 64
    stop: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.prompt is None and self.messages is None:
            raise ValueError(f"case {self.id!r}: give `prompt` or `messages`")
        if self.role not in ROLES:
            raise ValueError(f"case {self.id!r}: role must be one of {ROLES}")
        if isinstance(self.good, str):
            self.good = [self.good]
        if isinstance(self.bad, str):
            self.bad = [self.bad]
        if isinstance(self.stop, str):
            self.stop = [self.stop]
        if self.contrast is not None and not {"good", "bad"} <= set(self.contrast):
            raise ValueError(f"case {self.id!r}: contrast needs 'good' and 'bad'")
        for p in self.good + self.bad:
            re.compile(p)
        self.group = self.group or self.id

    @property
    def mode(self) -> str:
        """`contrast` (log-prob margin), `judged` (generation + rule) or `unchanged` (keep: equal to base)."""
        if self.contrast is not None:
            return "contrast"
        if self.good or self.bad or self.check:
            return "judged"
        return "unchanged"

    def text(self, model_dir=None) -> str:
        """The exact prompt text (rendering `messages` with the checkpoint's chat template if needed)."""
        if self.prompt is not None:
            return self.prompt + self.prefill
        from .prompts import render_chat

        return render_chat(self.messages, model_dir, prefill=self.prefill)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return {k: v for k, v in d.items() if v not in (None, [], {}, "") or k in ("id",)}


def load_cases(path: str | Path) -> list[Case]:
    """Read a JSONL case file. Ids must be unique."""
    names = {f.name for f in dataclasses.fields(Case)}
    out, seen = [], set()
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        d = json.loads(line)
        extra = set(d) - names
        if extra:
            raise ValueError(f"{path}:{n}: unknown fields {sorted(extra)}")
        c = Case(**d)
        if c.id in seen:
            raise ValueError(f"{path}:{n}: duplicate id {c.id!r}")
        seen.add(c.id)
        out.append(c)
    return out


def save_cases(cases: Iterable[Case], path: str | Path) -> None:
    Path(path).write_text("".join(json.dumps(c.to_dict(), ensure_ascii=False) + "\n" for c in cases))


def select(cases: Iterable[Case], split: str | None = None, role: str | None = None) -> list[Case]:
    return [c for c in cases if (split is None or c.split == split) and (role is None or c.role == role)]


def check_splits(cases: Iterable[Case], a: str = "tune", b: str = "holdout") -> list[str]:
    """Groups that appear in both splits (should be empty: a shared group leaks the holdout into tuning)."""
    cs = list(cases)
    ga = {c.group for c in cs if c.split == a}
    gb = {c.group for c in cs if c.split == b}
    return sorted(ga & gb)


_CHECKS: dict[str, Callable] = {}


def _check_fn(spec: str) -> Callable:
    if spec not in _CHECKS:
        mod, _, fn = spec.partition(":")
        if not fn:
            raise ValueError(f"check {spec!r} must look like 'module:function'")
        _CHECKS[spec] = getattr(importlib.import_module(mod), fn)
    return _CHECKS[spec]


def judge(text: str, case: Case) -> str:
    """Label one generation `good`, `bad` or `other` by the case's rule."""
    if case.check:
        r = _check_fn(case.check)(text, case)
        if r is True:
            return "good"
        if r is False:
            return "bad"
        if r not in LABELS:
            raise ValueError(f"check {case.check!r} returned {r!r}; expected one of {LABELS} or a bool")
        return r
    if any(re.search(p, text) for p in case.bad):
        return "bad"
    if any(re.search(p, text) for p in case.good):
        return "good"
    return "other"


def cut_at_stop(text: str, stops: Iterable[str]) -> str:
    """Cut `text` at the earliest stop string (the stop string itself is dropped)."""
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i >= 0:
            cut = min(cut, i)
    return text[:cut]
