"""Call records: one recorded agent tool call, replayable against the true file state.

A record (one JSON object per line in calls.jsonl) has at least

    key            unique id; the study's format is run|task|item|tc (the gate's unit is the task part)
    prompt         the exact text the served model saw before its turn
    prefix         the model's own text from its turn up to and including the opening delimiter of the argument to
                   regenerate (for edit_file: `...old_string:<|"|>`)
    old_string, new_string, filepath, allow_multiple    the recorded arguments
    file_sha       the file state the call ran against (text in files/<file_sha>.txt)
    error_kind     recorded tool outcome: ok | not_found | multiple | ...    strategy: the harness's match strategy
optionally task, repo, verified, state_uncertain, file_missing (the usable filter), and anything else.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

REQUIRED = ("key", "prompt", "prefix", "old_string", "filepath")


def usable(r: dict) -> bool:
    """The study's filter: verified prompt, certain file state, file present, non-empty recorded old_string."""
    return bool(r.get("verified", 1)) and not r.get("state_uncertain", 0) and not r.get("file_missing", 0) and bool(r.get("old_string"))


def iter_calls(path: str | Path, only_usable: bool = True) -> Iterator[dict]:
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                missing = [k for k in REQUIRED if k not in r]
                if missing:
                    raise ValueError(f"call {r.get('key')} lacks {missing}")
                if not only_usable or usable(r):
                    yield r


def load_calls(path: str | Path, keys: Iterable[str] | None = None, only_usable: bool = True) -> list[dict]:
    """Calls in key-file order if keys is given (every key must be usable), else file order."""
    rows = list(iter_calls(path, only_usable))
    if keys is None:
        return rows
    idx = {r["key"]: r for r in rows}
    want = list(keys)
    miss = [k for k in want if k not in idx]
    if miss:
        raise KeyError(f"{len(miss)} keys not usable, e.g. {miss[:3]}")
    return [idx[k] for k in want]


def read_keys(path: str | Path) -> list[str]:
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


class FileStore:
    """File states by sha: DIR/<sha>.txt, cached in memory."""

    def __init__(self, files_dir: str | Path):
        self.dir = Path(files_dir)
        self._c: dict[str, str] = {}

    def __call__(self, call: dict) -> str:
        sha = call["file_sha"]
        if sha not in self._c:
            self._c[sha] = (self.dir / f"{sha}.txt").read_text()
        return self._c[sha]


def task_of(call: dict) -> str:
    if call.get("task"):
        return call["task"]
    if call.get("instance_id"):
        return call["instance_id"]
    parts = call["key"].split("|")
    return parts[1] if len(parts) > 1 else call["key"]
