"""A tiny, randomly initialised Gemma 4 checkpoint for tests and quickstarts (no download, no real weights).

    from weightprog.testing import write_tiny_model
    d = write_tiny_model("/tmp/tiny_g4", seed=0)     # config.json, model.safetensors, tokenizer.json

The config mirrors Gemma 4 E4B's structure at toy size and exercises every feature the backends must reproduce:
sliding-window and global layers (different head dims, proportional partial RoPE on the global ones), KV-sharing
layers without k/v projections, per-layer input embeddings, a non-trivial per-layer scalar, tied embeddings and a
final logit softcap small enough to bend the logits. The tokenizer is character-level (printable ASCII, newline,
tab) plus Gemma's string delimiter `<|"|>` and a few control tokens as special tokens.

The weights are written in the Hugging Face layout of Gemma4ForConditionalGeneration (`model.language_model.*`), so
`mlx_lm.utils.load_model` and `transformers.Gemma4ForConditionalGeneration.from_pretrained` both load the directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..io.safetensors_np import save_file

SPECIALS = ["<|\"|>", "<|tool_call>", "<tool_call|>", "<|turn>", "<turn|>", "<|channel>", "<channel|>"]


def tiny_text_config(vocab_size: int, **over) -> dict:
    t = {
        "attention_bias": False, "attention_dropout": 0.0, "attention_k_eq_v": False, "bos_token_id": 2,
        "dtype": "float32", "enable_moe_block": False, "eos_token_id": 1, "final_logit_softcapping": 6.0,
        "global_head_dim": 32, "head_dim": 16, "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 64,
        "hidden_size_per_layer_input": 8, "initializer_range": 0.02, "intermediate_size": 96,
        "layer_types": ["sliding_attention", "sliding_attention", "full_attention",
                        "sliding_attention", "sliding_attention", "full_attention"],
        "max_position_embeddings": 4096, "model_type": "gemma4_text", "moe_intermediate_size": None,
        "num_attention_heads": 4, "num_experts": None, "num_global_key_value_heads": None, "num_hidden_layers": 6,
        "num_key_value_heads": 2, "num_kv_shared_layers": 2, "pad_token_id": 0,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0,
                                               "rope_type": "proportional"},
                            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
        "sliding_window": 8, "tie_word_embeddings": True, "top_k_experts": None,
        "use_bidirectional_attention": None, "use_cache": True, "use_double_wide_mlp": False,
        "vocab_size": vocab_size, "vocab_size_per_layer_input": vocab_size,
    }
    t.update(over)
    return t


def tiny_config(vocab_size: int, **over) -> dict:
    return {"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4", "vocab_size": vocab_size,
            "text_config": tiny_text_config(vocab_size, **over), "vision_config": None, "audio_config": None,
            "tie_word_embeddings": True, "eos_token_id": [1], "dtype": "float32", "initializer_range": 0.02}


def build_tokenizer():
    """Character-level `tokenizers.Tokenizer`: <pad>=0, <eos>=1, <bos>=2, <unk>=3, characters, then specials."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}
    for c in [chr(i) for i in range(32, 127)] + ["\n", "\t"]:
        vocab[c] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r".|\n"), behavior="isolated")
    tok.decoder = decoders.Fuse()
    tok.add_special_tokens(SPECIALS)
    return tok


def tiny_weights(text_cfg: dict, seed: int = 0, scale: float = 1.0) -> dict[str, np.ndarray]:
    """Random float32 weights with the HF Gemma4ForConditionalGeneration names."""
    rng = np.random.default_rng(seed)
    t = text_cfg
    H, I, V = t["hidden_size"], t["intermediate_size"], t["vocab_size"]
    nH, nKV, P, n = t["num_attention_heads"], t["num_key_value_heads"], t["hidden_size_per_layer_input"], t["num_hidden_layers"]
    first_shared = n - t["num_kv_shared_layers"]
    pre = "model.language_model."

    def lin(o, i, s=1.0):
        return (rng.standard_normal((o, i)) * s * scale / np.sqrt(i)).astype(np.float32)

    def norm(d, s=0.15):
        return (1.0 + s * rng.standard_normal(d)).astype(np.float32)

    w = {pre + "embed_tokens.weight": (rng.standard_normal((V, H)) * 0.5).astype(np.float32),
         pre + "norm.weight": norm(H, 0.3) * 2.0}
    if P:
        w[pre + "embed_tokens_per_layer.weight"] = (rng.standard_normal((t["vocab_size_per_layer_input"], n * P)) * 0.3).astype(np.float32)
        w[pre + "per_layer_model_projection.weight"] = lin(n * P, H)
        w[pre + "per_layer_projection_norm.weight"] = norm(P)
    for i, typ in enumerate(t["layer_types"]):
        hd = t["global_head_dim"] if typ == "full_attention" else t["head_dim"]
        L = f"{pre}layers.{i}."
        w[L + "self_attn.q_proj.weight"] = lin(nH * hd, H)
        w[L + "self_attn.q_norm.weight"] = norm(hd)
        if i < first_shared:
            w[L + "self_attn.k_proj.weight"] = lin(nKV * hd, H)
            w[L + "self_attn.v_proj.weight"] = lin(nKV * hd, H)
            w[L + "self_attn.k_norm.weight"] = norm(hd)
        w[L + "self_attn.o_proj.weight"] = lin(H, nH * hd)
        w[L + "mlp.gate_proj.weight"] = lin(I, H)
        w[L + "mlp.up_proj.weight"] = lin(I, H)
        w[L + "mlp.down_proj.weight"] = lin(H, I)
        for nm in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            w[L + nm + ".weight"] = norm(H)
        if P:
            w[L + "per_layer_input_gate.weight"] = lin(P, H)
            w[L + "per_layer_projection.weight"] = lin(H, P)
            w[L + "post_per_layer_input_norm.weight"] = norm(H)
        w[L + "layer_scalar"] = np.array([rng.uniform(0.4, 1.0)], dtype=np.float32)
    return w


def write_tiny_model(out_dir: str | Path, seed: int = 0, dtype: str = "F32", **config_over) -> Path:
    """Write config.json, model.safetensors (dtype "F32" or "BF16") and tokenizer.json; return the directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = build_tokenizer()
    vocab = tok.get_vocab_size()
    cfg = tiny_config(vocab, **config_over)
    if dtype == "BF16":
        cfg["dtype"] = cfg["text_config"]["dtype"] = "bfloat16"
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    tok.save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
                                                           "unk_token": "<unk>", "tokenizer_class": "PreTrainedTokenizerFast"}))
    save_file(tiny_weights(cfg["text_config"], seed=seed), out / "model.safetensors", default_dtype=dtype,
              metadata={"format": "pt"})
    return out
