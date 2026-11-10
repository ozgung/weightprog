---
name: gemma4-internals
description: Architecture facts about Gemma 4 text models (E4B in detail) that matter when reading activations or editing weights - layer types and head dims, KV sharing, per-layer inputs, layer scalar, norms, tied unembedding with softcap, tool-call format, and parameter names in HF and MLX. Use before capturing activations, choosing modules to edit or deriving logit-space directions for Gemma 4.
---

# Gemma 4 internals for weight editing

Facts are from the E4B `config.json`, the mlx_lm `gemma4_text.py` and transformers `modeling_gemma4.py` code, and
the weightprog case study. `weightprog.names.ModelSpec.from_config(dir)` computes the structural ones for any
Gemma 4 text config.

## E4B shape

- 42 decoder layers, hidden 2560, MLP intermediate 10240 (no double-wide MLP on E4B), 8 query heads, 2 KV heads,
  vocabulary 262,144, about 7.9B parameters in bf16 (2,076 tensors; the per-layer embedding table [262144, 10752]
  is about 2.8B of them).
- **Layer types**: 5 sliding-window layers then 1 global, repeated; global layers are 5, 11, 17, 23, 29, 35, 41.
  Sliding window 512 (a query at p sees keys p-511..p). Head dim 256 on sliding layers, 512 on global layers, where
  RoPE ("proportional", theta 1e6) rotates only the first quarter of each head; sliding layers use default RoPE,
  theta 1e4. Attention scale is 1.0 (queries and keys are RMS-normalised per head; values are normalised without a
  weight).
- **KV sharing**: the last 18 layers (24-41) have no k_proj / v_proj. Each reuses the keys and values of the last
  non-shared layer of its own type: sliding layers read layer 22, global layers read layer 23. Consequence: an edit
  in layers 24-41 changes only the computation at the position being generated, never what other positions read.
  Editable modules: 24 x 7 + 18 x 5 = 258.

## One decoder layer (both frameworks)

```
h_attn = h + post_attention_layernorm(o_proj(attn(input_layernorm(h))))
y      = gelu_tanh(gate_proj(x)) * up_proj(x),   x = pre_feedforward_layernorm(h_attn)    # "mlp" capture = y
h_mlp  = h_attn + post_feedforward_layernorm(down_proj(y))                                # "resid_mid" = h_attn
h_pl   = h_mlp + post_per_layer_input_norm(per_layer_projection(gelu_tanh(per_layer_input_gate(h_mlp)) * pli_i))
out    = h_pl * layer_scalar                                                              # "resid" = out
logits = softcap(E @ norm(out_last)),  softcap(z) = 30 tanh(z / 30)
```

- **RMSNorm multiplies by its weight directly** (`x / rms(x) * w`, no `1 + w`), computed in float32 in HF.
- **layer_scalar** is a per-layer scalar buffer (0.48 on layer 41). Residual captures are post-scalar; the MLP output
  enters pre-scalar. For the last layer the scalar cancels against the final norm's rms in the logits, but any
  hand simulation that adds a pre-scalar change to a post-scalar residual must multiply it by s.
- **Per-layer inputs** (`pli_i`, width 256): `(per_layer_model_projection(h0) / sqrt(H), normed) +
  embed_tokens_per_layer(ids) * sqrt(256)`, times 2^-1/2; h0 is the scaled token embedding.
- **Embeddings** are scaled by sqrt(hidden) on input and tied to the output (`lm_head` = `embed_tokens`).
- Final logit softcap 30. A logit-space direction in down_proj output space for layer L must pass through
  `w_post(L)`, the residual, `w_final` and `E`; see `weightprog.construct.logit_direction`.

## Names

| Framework | Prefix |
|---|---|
| HF checkpoint / torch (Gemma4ForConditionalGeneration) | `model.language_model.layers.{i}.mlp.down_proj.weight` |
| MLX (mlx_lm gemma4) | `language_model.model.layers.{i}.mlp.down_proj.weight` |
| text-only Gemma4ForCausalLM | `model.layers.{i}.mlp.down_proj.weight` |

weightprog's canonical names are the HF checkpoint's; `layer_scalar` has no `.weight` suffix.

## Tool calls (if your target involves the model's tool use)

Gemma 4 renders a call as `<|tool_call>call:NAME{k1:v1,k2:v2}<tool_call|>`, keys in sorted order, strings wrapped
in the single token `<|"|>` with no escaping inside, booleans `true`/`false`. So `edit_file` arguments come as
`filepath`, `new_string`, `old_string`: new_string is written before old_string (727 of 730 search calls and 320 of
320 gate calls in the study). Tool responses in the study's harness were JSON-escaped inside a string, so file text
reached the model with `\n` and `\"` escapes; the model's own thought and new_string then often carried escaped
code, and old_string copied it (the escape artefact).

## Numerics

- bf16 on MLX: the manual capture forward and the native cached forward differ by up to about 0.5 nats on
  14k-token prompts (0.04 in a short smoke test); prefer native log-probs for scoring.
- MLX vs HF torch on the same checkpoint: greedy text identical 39/40 on 40 gate calls in the study's port check,
  log-probs within 0.12 nats. Sampled draws differ by RNG.
- The QAT checkpoint's `per_layer_model_projection` weights are about 1/64 the magnitude of the standard E4B's.
