"""Gemma 4 tool-call string codec.

Gemma 4's chat template renders a tool call as

    <|tool_call>call:NAME{key1:VALUE1,key2:VALUE2}<tool_call|>

with the arguments in sorted key order (so edit_file writes filepath, new_string, old_string in that order), strings
wrapped in the delimiter token `<|"|>` (no escaping inside), booleans as true/false, mappings as {k:v} with bare keys,
lists as [a,b] and other values verbatim. A replay "prefix" is the model's own text up to and including the opening
delimiter of the argument to regenerate, e.g. `...old_string:<|"|>`; generation stops at the closing `<|"|>`.
"""

from __future__ import annotations

from typing import Any

Q = '<|"|>'
CALL_OPEN = "<|tool_call>call:"
CALL_CLOSE = "<tool_call|>"


def encode_value(v: Any) -> str:
    if isinstance(v, str):
        if Q in v:
            raise ValueError("a string argument cannot contain the delimiter token")
        return Q + v + Q
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}:{encode_value(v[k])}" for k in sorted(v)) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(encode_value(x) for x in v) + "]"
    return str(v)


def encode_call(name: str, args: dict) -> str:
    return CALL_OPEN + name + "{" + ",".join(f"{k}:{encode_value(args[k])}" for k in sorted(args)) + "}" + CALL_CLOSE


def arg_order(args: dict) -> list[str]:
    return sorted(args)


def call_prefix(name: str, args: dict, upto: str) -> str:
    """The call text up to and including `upto:<|"|>` (the arguments before `upto` in sorted order are written)."""
    if upto not in args:
        raise KeyError(upto)
    head = CALL_OPEN + name + "{"
    parts = []
    for k in sorted(args):
        if k == upto:
            return head + "".join(p + "," for p in parts) + f"{k}:" + Q
        parts.append(f"{k}:{encode_value(args[k])}")
    raise AssertionError("unreachable")


def separator(next_arg: str) -> str:
    """Text between a closed string argument and the opening delimiter of the next one: <|"|>,next:<|"|>."""
    return Q + "," + next_arg + ":" + Q


class _Parser:
    def __init__(self, s: str, i: int = 0):
        self.s, self.i = s, i

    def peek(self, t: str) -> bool:
        return self.s.startswith(t, self.i)

    def eat(self, t: str) -> None:
        if not self.peek(t):
            raise ValueError(f"expected {t!r} at {self.i}: {self.s[self.i:self.i + 30]!r}")
        self.i += len(t)

    def value(self):
        if self.peek(Q):
            self.i += len(Q)
            j = self.s.find(Q, self.i)
            if j < 0:
                raise ValueError("unterminated string")
            v = self.s[self.i:j]
            self.i = j + len(Q)
            return v
        if self.peek("{"):
            return self.mapping()
        if self.peek("["):
            self.eat("[")
            out = []
            while not self.peek("]"):
                out.append(self.value())
                if self.peek(","):
                    self.eat(",")
            self.eat("]")
            return out
        j = self.i
        while j < len(self.s) and self.s[j] not in ",}]":
            j += 1
        raw, self.i = self.s[self.i:j], j
        if raw == "true":
            return True
        if raw == "false":
            return False
        for cast in (int, float):
            try:
                return cast(raw)
            except ValueError:
                pass
        return raw

    def mapping(self) -> dict:
        self.eat("{")
        out = {}
        while not self.peek("}"):
            j = self.s.index(":", self.i)
            key = self.s[self.i:j]
            if key.startswith(Q) and key.endswith(Q):
                key = key[len(Q):-len(Q)]
            self.i = j + 1
            out[key] = self.value()
            if self.peek(","):
                self.eat(",")
        self.eat("}")
        return out


def decode_call(text: str) -> tuple[str, dict]:
    """Parse the first `<|tool_call>call:NAME{...}` in text; returns (name, args)."""
    i = text.find(CALL_OPEN)
    if i < 0:
        raise ValueError("no tool call")
    i += len(CALL_OPEN)
    j = text.index("{", i)
    name = text[i:j]
    p = _Parser(text, j)
    return name, p.mapping()


def split_prefix(record: dict, arg: str = "new_string", next_arg: str = "old_string") -> str:
    """Full-call prefix of a frozen-prefix replay record: its prefix (which ends with `next_arg:<|"|>`) cut right
    after `arg:<|"|>`. Raises ValueError unless the recorded `arg` value and the separator follow the cut verbatim."""
    mark = arg + ":" + Q
    pre = record["prefix"]
    i = pre.rfind(mark)
    if i < 0 or pre[i + len(mark):] != (record.get(arg) or "") + separator(next_arg):
        raise ValueError(f"prefix of {record.get('key')} cannot be cut at {arg}")
    return pre[: i + len(mark)]
