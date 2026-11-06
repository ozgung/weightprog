"""Synthetic edit_file call records and file states (no third-party data), for tests and the quickstart.

    calls, files = synthetic_calls(n=8, seed=0)      # list of call dicts, {sha: file text}
    write_call_set(out_dir, calls, files)            # calls.jsonl, files/<sha>.txt, usable_keys.txt

Each call has a toy prompt, a prefix that writes `filepath` and `new_string` and opens `old_string`, a recorded
old_string taken from the file (some calls carry an escape artefact: a literal backslash-n where the file has a
newline, like the defect studied), and the outcome an exact matcher would give.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from ..replay.codec import Q, call_prefix

FUNCS = ["def add(a, b):\n    return a + b\n", "def neg(x):\n    return -x\n", "def greet(n):\n    print(\"hi\", n)\n",
         "class Box:\n    def __init__(self, v):\n        self.v = v\n", "def first(xs):\n    return xs[0]\n",
         "def lines(s):\n    return s.split(\"\\n\")\n"]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def synthetic_calls(n: int = 8, seed: int = 0, artefact_rate: float = 0.4, n_tasks: int | None = None):
    rng = random.Random(seed)
    calls, files = [], {}
    n_tasks = n_tasks or max(1, n // 2)
    for i in range(n):
        t = i % n_tasks
        body = "".join(rng.sample(FUNCS, 3))
        path = f"pkg/mod{t}.py"
        files[_sha(body)] = body
        start = body.index("def") if "def" in body else 0
        old = body[start: body.index("\n", start) + 1 + body[body.index("\n", start) + 1:].index("\n")]
        new = old.replace("return", "return  ")
        escaped = rng.random() < artefact_rate
        rec_old = old.replace("\n", "\\n") if escaped else old
        args = {"filepath": path, "new_string": new.replace("\n", "\\n") if escaped else new, "old_string": rec_old}
        thought = f"I will edit `{path}`.\n"
        prefix = "<|channel>thought\n" + thought + "<channel|>" + call_prefix("edit_file", args, "old_string")
        count = body.count(rec_old)
        kind = "ok" if count == 1 else ("not_found" if count == 0 else "multiple")
        calls.append({"key": f"syn|task{t}|{i}|0", "task": f"task{t}", "repo": f"org/repo{t % 3}", "filepath": path,
                      "prompt": f"<bos><|turn>user\nFix the function in {path}.<turn|>\n<|turn>model\n", "prefix": prefix,
                      "old_string": rec_old, "new_string": args["new_string"], "allow_multiple": False,
                      "file_sha": _sha(body), "error_kind": kind, "strategy": "exact" if kind == "ok" else None,
                      "verified": 1, "state_uncertain": 0, "file_missing": 0})
    return calls, files


def write_call_set(out_dir: str | Path, calls: list[dict], files: dict[str, str]) -> Path:
    out = Path(out_dir)
    (out / "files").mkdir(parents=True, exist_ok=True)
    with (out / "calls.jsonl").open("w") as f:
        for c in calls:
            f.write(json.dumps(c) + "\n")
    for sha, text in files.items():
        (out / "files" / f"{sha}.txt").write_text(text)
    (out / "usable_keys.txt").write_text("\n".join(c["key"] for c in calls) + "\n")
    return out


assert Q  # the delimiter is part of every prefix built here
