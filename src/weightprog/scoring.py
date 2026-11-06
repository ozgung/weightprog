"""Pluggable scoring of regenerated tool-call arguments against the true file state.

    from weightprog.scoring import EditScorer, escape_artefact, exact_copy, compiles
    scorer = EditScorer(replacer=ExactReplacer())          # or SwegemmaReplacer() if swegemma is installed
    row = scorer.score(call, gen_old)                       # frozen-prefix row: outcome, accepted, exact_copy, ...
    row = scorer.score_fullcall(call, gen_old, gen_new)     # full-call row: also compiles with the generated new

Built-in metrics:
- escape_artefact(text, file): the text contains a literal backslash-n or backslash-quote that the file does not
  contain (the study's definition; text that really has a backslash in the file is not counted).
- exact_copy(text, file): non-empty and occurs verbatim in the file.
- compiles(text, path): compile() of a .py file's new content (None for other files).

Replacement semantics are pluggable because they belong to the agent harness. `ExactReplacer` accepts an
old_string that occurs exactly once (or at least once with allow_multiple). The study scored with its harness's
`swegemma.edit.apply_replacement`, which also tries unescaped, whitespace-flexible and regex matches; that function
is not part of this library. `SwegemmaReplacer` calls it when the package is importable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

ESCAPES = ("\\n", '\\"')


def escape_artefact(text: str, file_text: str, escapes=ESCAPES) -> int:
    return int(any(e in text and e not in file_text for e in escapes))


def exact_copy(text: str, file_text: str) -> int:
    return int(bool(text) and text in file_text)


def compiles(text: str | None, path: str) -> int | None:
    if text is None or not path.endswith(".py"):
        return None
    try:
        compile(text, path, "exec")
        return 1
    except (SyntaxError, ValueError):
        return 0


@dataclass
class Replacement:
    outcome: str            # applied_<strategy> | not_found | multiple | empty | other_error
    new_content: str | None


class Replacer(Protocol):
    def __call__(self, content: str, old: str, new: str, path: str, allow_multiple: bool) -> Replacement: ...


class ExactReplacer:
    """Exact-match str.replace: unique match required unless allow_multiple."""

    def __call__(self, content, old, new, path, allow_multiple=False) -> Replacement:
        if not old:
            return Replacement("empty", None)
        n = content.count(old)
        if n == 0:
            return Replacement("not_found", None)
        if n > 1 and not allow_multiple:
            return Replacement("multiple", None)
        return Replacement("applied_exact", content.replace(old, new) if allow_multiple else content.replace(old, new, 1))


class SwegemmaReplacer:
    """Adapter for the competition harness's edit function (`swegemma.edit.apply_replacement`), if installed.
    Outcome mapping as in the study's scorer: ValueError mentioning "not found" -> not_found; "occurs" or
    "multiple" -> multiple; any other exception -> other_error; success -> applied_<result.strategy>."""

    def __init__(self):
        from swegemma.edit import apply_replacement  # noqa: F401  (ImportError if the harness is absent)

        self._f = apply_replacement

    def __call__(self, content, old, new, path, allow_multiple=False) -> Replacement:
        if not old:
            return Replacement("empty", None)
        try:
            res = self._f(content, old, new, filepath=path, allow_multiple=allow_multiple)
        except ValueError as e:
            m = str(e)
            return Replacement("not_found" if "not found" in m else "multiple" if ("occurs" in m or "multiple" in m)
                               else "other_error", None)
        except Exception:  # noqa: BLE001
            return Replacement("other_error", None)
        return Replacement(f"applied_{res.strategy}", res.new_content)


def default_replacer() -> Replacer:
    try:
        return SwegemmaReplacer()
    except ImportError:
        return ExactReplacer()


class EditScorer:
    """Scores regenerated edit_file arguments. `file_text(call)` returns the true file state of a call record."""

    def __init__(self, replacer: Replacer | None = None, file_text: Callable[[dict], str] | None = None,
                 extra_metrics: dict[str, Callable[[str, str], int]] | None = None):
        self.replacer = replacer or ExactReplacer()
        self.file_text = file_text or (lambda call: call["file_text"])
        self.extra = extra_metrics or {}

    def score(self, call: dict, gen: str, closed: int = 1) -> dict:
        """Frozen-prefix row: the generated old_string with the call's recorded new_string."""
        content = self.file_text(call)
        new = call.get("new_string") or ""
        rep = self.replacer(content, gen, new, call["filepath"], bool(call.get("allow_multiple")))
        rec = self.replacer(content, call["old_string"], new, call["filepath"], bool(call.get("allow_multiple")))
        row = {"key": call["key"], "recorded_kind": call.get("error_kind"), "recorded_strategy": call.get("strategy"),
               "recorded_replayed": rec.outcome, "outcome": rep.outcome, "accepted": int(rep.outcome.startswith("applied")),
               "exact_copy": exact_copy(gen, content), "same_as_recorded": int(gen == call["old_string"]), "closed": closed,
               "compiles": compiles(rep.new_content, call["filepath"]),
               "same_result": (int(rep.new_content == rec.new_content) if (rep.new_content is not None and rec.new_content is not None) else None),
               "escape_artefact": escape_artefact(gen, content)}
        for k, f in self.extra.items():
            row[k] = f(gen, content)
        return row

    def score_fullcall(self, call: dict, gen_old: str, gen_new: str, closed: int = 1, closed_new: int = 1) -> dict:
        """Full-call row: the generated pair, plus the registered (frozen new_string) compile check."""
        content = self.file_text(call)
        path, multi = call["filepath"], bool(call.get("allow_multiple"))
        rep = self.replacer(content, gen_old, gen_new, path, multi)
        frz = self.replacer(content, gen_old, call.get("new_string") or "", path, multi)
        return {"key": call["key"], "recorded_kind": call.get("error_kind"), "recorded_strategy": call.get("strategy"),
                "outcome": rep.outcome, "accepted": int(rep.outcome.startswith("applied")),
                "exact_copy": exact_copy(gen_old, content), "closed_new": closed_new, "closed": closed,
                "compiles": compiles(rep.new_content, path),
                "artefact_old": escape_artefact(gen_old, content), "artefact_new": escape_artefact(gen_new, content),
                "same_new_as_recorded": int(gen_new == (call.get("new_string") or "")),
                "compiles_frozen_new": compiles(frz.new_content, path)}
