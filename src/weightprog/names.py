"""Canonical tensor names and the architecture facts editing needs, read from a Gemma 4 config.json.

Canonical names follow the Hugging Face checkpoint of Gemma4ForConditionalGeneration:

    model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    model.language_model.layers.{i}.mlp.{gate,up,down}_proj.weight
    model.language_model.{embed_tokens,norm}.weight, ...layers.{i}.post_feedforward_layernorm.weight, ...layer_scalar

Each backend maps them to its own parameter names (MLX: `language_model.model.layers...`; a text-only
Gemma4ForCausalLM checkpoint: `model.layers...`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

PREFIX = "model.language_model."
ATTN_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJ = ("gate_proj", "up_proj", "down_proj")
EDITABLE_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$")


def module_name(layer: int, proj: str) -> str:
    """Canonical weight name of a decoder-layer linear, e.g. module_name(41, "down_proj")."""
    if proj in ATTN_PROJ:
        return f"{PREFIX}layers.{layer}.self_attn.{proj}.weight"
    if proj in MLP_PROJ:
        return f"{PREFIX}layers.{layer}.mlp.{proj}.weight"
    raise ValueError(f"unknown projection {proj!r}")


def parse_module(name: str) -> tuple[int, str]:
    """(layer, proj) of a canonical editable name."""
    m = EDITABLE_RE.match(name)
    if not m:
        raise ValueError(f"not a canonical decoder-layer linear: {name!r}")
    return int(m.group(1)), m.group(2).split(".")[1]


def canonical(name: str) -> str:
    """Accept a few spellings (MLX's, a text-only checkpoint's, without .weight) and return the canonical name."""
    n = name
    for p in ("language_model.model.", "model.language_model.", "language_model.", "model."):
        if n.startswith(p):
            n = n[len(p):]
            break
    if not (n.endswith(".weight") or n.endswith("layer_scalar")):
        n += ".weight"
    return PREFIX + n


def file_stem(name: str) -> str:
    """The edit-file stem used by the original kernel: layers_41_mlp_down_proj."""
    return name.removeprefix(PREFIX).removesuffix(".weight").replace(".", "_")


@dataclass
class ModelSpec:
    """Architecture facts of a Gemma 4 text model (from config.json; no weights needed)."""

    n_layers: int
    hidden: int
    intermediate: list[int]
    n_heads: int
    n_kv_heads: int
    head_dim: int
    global_head_dim: int
    layer_types: list[str]
    sliding_window: int
    n_kv_shared: int
    hidden_per_layer: int
    vocab: int
    softcap: float | None
    tie_embeddings: bool = True
    rms_eps: float = 1e-6
    kv_source: list[int] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: dict | str | Path) -> "ModelSpec":
        if not isinstance(cfg, dict):
            p = Path(cfg)
            cfg = json.loads((p / "config.json" if p.is_dir() else p).read_text())
        t = cfg.get("text_config") or cfg
        n = t["num_hidden_layers"]
        types = t.get("layer_types") or [("full_attention" if (i + 1) % 6 == 0 else "sliding_attention") for i in range(n)]
        shared = int(t.get("num_kv_shared_layers") or 0)
        first_shared = n - shared
        wide = bool(t.get("use_double_wide_mlp"))
        inter = [t["intermediate_size"] * (2 if (wide and i >= first_shared > 0) else 1) for i in range(n)]
        src = list(range(n))
        if shared:
            last = {}
            for i in range(first_shared):
                last[types[i]] = i
            for j in range(first_shared, n):
                src[j] = last[types[j]]
        return cls(n_layers=n, hidden=t["hidden_size"], intermediate=inter, n_heads=t["num_attention_heads"],
                   n_kv_heads=t["num_key_value_heads"], head_dim=t["head_dim"],
                   global_head_dim=t.get("global_head_dim") or t["head_dim"], layer_types=types,
                   sliding_window=t["sliding_window"], n_kv_shared=shared,
                   hidden_per_layer=int(t.get("hidden_size_per_layer_input") or 0), vocab=t["vocab_size"],
                   softcap=t.get("final_logit_softcapping"), tie_embeddings=bool(cfg.get("tie_word_embeddings", t.get("tie_word_embeddings", True))),
                   rms_eps=float(t.get("rms_norm_eps", 1e-6)), kv_source=src)

    @property
    def first_kv_shared(self) -> int:
        return self.n_layers - self.n_kv_shared

    def has_kv(self, layer: int) -> bool:
        return layer < self.first_kv_shared

    def layer_head_dim(self, layer: int) -> int:
        return self.global_head_dim if self.layer_types[layer] == "full_attention" else self.head_dim

    def is_sliding(self, layer: int) -> bool:
        return self.layer_types[layer] == "sliding_attention"

    def module_shape(self, name: str) -> tuple[int, int]:
        """[out, in] of an editable module, from the config alone."""
        layer, proj = parse_module(name)
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer {layer} out of range")
        hd, H = self.layer_head_dim(layer), self.hidden
        if proj in ("k_proj", "v_proj") and not self.has_kv(layer):
            raise ValueError(f"layer {layer} is KV-shared: it has no {proj}")
        return {"q_proj": (self.n_heads * hd, H), "k_proj": (self.n_kv_heads * hd, H), "v_proj": (self.n_kv_heads * hd, H),
                "o_proj": (H, self.n_heads * hd), "gate_proj": (self.intermediate[layer], H),
                "up_proj": (self.intermediate[layer], H), "down_proj": (H, self.intermediate[layer])}[proj]

    def editable_modules(self) -> list[str]:
        """Every decoder-layer linear that exists in this model (E4B: 258)."""
        out = []
        for i in range(self.n_layers):
            for p in ATTN_PROJ:
                if p in ("k_proj", "v_proj") and not self.has_kv(i):
                    continue
                out.append(module_name(i, p))
            out += [module_name(i, p) for p in MLP_PROJ]
        return out

    def is_editable(self, name: str) -> bool:
        try:
            self.module_shape(name)
            return True
        except ValueError:
            return False
