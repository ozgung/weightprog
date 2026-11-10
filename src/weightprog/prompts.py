"""Turn chat messages into the exact text a Gemma 4 checkpoint was trained on.

    from weightprog.prompts import render_chat
    text = render_chat([{"role": "user", "content": "Write a function ..."}], model_dir)
    ids = backend.tokenizer.encode(text)       # specials such as <bos> and <|turn> are single tokens

The checkpoint's own `chat_template.jinja` (or the `chat_template` field of `tokenizer_config.json`) is rendered with
jinja2 when both are available. Without them, a minimal Gemma 4 layout is used:
`<bos><|turn>user\\n...<turn|>\\n<|turn>model\\n` (system turns as `<|turn>system`). Thinking is off unless asked for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

BOS, SOT, EOT = "<bos>", "<|turn>", "<turn|>"


def _template(model_dir: str | Path | None) -> str | None:
    if model_dir is None:
        return None
    d = Path(model_dir)
    if (d / "chat_template.jinja").exists():
        return (d / "chat_template.jinja").read_text()
    cfg = d / "tokenizer_config.json"
    if cfg.exists():
        t = json.loads(cfg.read_text()).get("chat_template")
        if isinstance(t, str):
            return t
    return None


def render_chat(messages: Sequence[dict], model_dir: str | Path | None = None, add_generation_prompt: bool = True,
                enable_thinking: bool = False, template: str | None = None, prefill: str = "") -> str:
    """Render `messages` ([{"role": "system"|"user"|"assistant", "content": str}, ...]) to prompt text.

    `prefill` is appended after the generation prompt: the start of the model's answer, e.g. a code fence, so the
    behaviour you measure starts within a few tokens."""
    tpl = template or _template(model_dir)
    if tpl is not None:
        try:
            import jinja2

            env = jinja2.Environment()
            text = env.from_string(tpl).render(messages=list(messages), bos_token=BOS,
                                               add_generation_prompt=add_generation_prompt,
                                               enable_thinking=enable_thinking)
            return text + prefill
        except ImportError:
            import warnings
            warnings.warn("jinja2 is not installed: using weightprog's minimal Gemma 4 turn layout, not the checkpoint's "
                          "chat template", stacklevel=2)
    out = [BOS]
    for m in messages:
        role = {"assistant": "model"}.get(m["role"], m["role"])
        out.append(f"{SOT}{role}\n{m['content']}{EOT}\n")
    if add_generation_prompt:
        out.append(f"{SOT}model\n")
    return "".join(out) + prefill
