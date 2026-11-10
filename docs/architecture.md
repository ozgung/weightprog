# Architecture

weightprog has three layers: **instruments** (backends) that run the model the same way every time, **edits** (plain
files: low-rank module additions and their matched random controls), and **judgement** (cases, evaluation, side
effects) that compares arms on the same prompts and seeds.

```
  cases.jsonl (target / keep, tune / holdout, groups, rules)
        |
        v
  evaluate.run(backend, cases, arms)  ->  rows (arm, case, draw, text, label | margin)
        |                                      |
        |   probe / construct  -> edit/*.npz   v
        |   controls.matched_control  ->  evaluate.compare(rows, side_effects(...), Criteria)  ->  report + verdict
        v
  backends (mlx | torch): load, edits() context, generate, logprobs, capture, read_weight
```

## Backends

`weightprog.backends.base.Backend` is the only interface the rest of the library uses.

| Method | Semantics |
|---|---|
| `load(model_dir, dtype, device)` | `config.json`, `model*.safetensors`, `tokenizer.json`. MLX: `mlx_lm.utils.load_model(strict=True)`, cast with `set_dtype`. Torch: `Gemma4ForConditionalGeneration` (or `Gemma4ForCausalLM` for text-only checkpoints), vision/audio towers dropped after load. |
| `tokenizer` | `tokenizers.Tokenizer` wrapper: no special tokens added, specials decoded verbatim. Shared by both backends, so token ids are identical. |
| `spec` | `names.ModelSpec` from config: layer types, head dims, KV-sharing sources, softcap, editable modules. |
| `edits(edit_set)` | context manager: `W' = (W.float32 + B @ A).astype(W.dtype)`; originals kept and restored bit for bit on exit (also on error). Nesting works. |
| `generate(ids, max_new, draws, seeds, stop_ids, sampling)` | one Generate segment of `decode_segments`. |
| `decode_segments(ids, [Generate(max_new, stops) \| Force(ids)], draws, seeds)` | prefill `ids[:-1]` once (chunks of 2048), copy the cache per draw, feed `ids[-1]`; draw 0 greedy, draws >= 1 sampled. A stop token ends a Generate segment and is not fed; a Force segment's tokens are fed before the next Generate. |
| `logprobs(ids, positions)` | native cached forward (the generation code path), log-softmax over the vocabulary; row j predicts token `positions[j] + 1`. |
| `capture(ids, positions, layers, what, attn_layers)` | manual forward over the same modules. `resid` (after the layer, post layer-scalar), `resid_pre`, `resid_mid` (what the MLP output is added to), `mlp` (GeGLU activations entering down_proj), `mlp_out`, `pli` (per-layer input), `attn` (probabilities with the sliding window and KV sharing), `logits` (softcapped, float32). |
| `read_weight(name)` | float32 numpy copy of any parameter or buffer by canonical name (`model.language_model.layers.{i}...`). |

**Sampling.** Both backends implement mlx_lm's order: top-p on the untempered distribution, then top-k, then
categorical at temperature T (served settings T 0.6, top_p 0.95, top_k 64). The RNGs differ, so sampled draws are
not token-identical across backends; greedy agrees up to numerics. Never mix generations from different backends in
one paired comparison.

**Parity tests** (`tests/test_parity.py`, `tests/test_backends.py`) build a tiny random checkpoint with every Gemma 4
feature that matters (sliding and global layers, proportional partial RoPE, KV-shared layers without k/v
projections, per-layer inputs, a layer scalar, softcap, tied embeddings) and check both backends against each other
and against `mlx_lm`'s own decode loop.

## Edits and controls

An edit is a directory of `layers_{i}_{block}_{proj}.npz` files, each with `module` (canonical name), `B [out, r]`,
`A [r, in]` (float32) and `meta` (JSON). Validation: allowed module (decoder-layer q/k/v/o/gate/up/down that exists
in this model), shapes, 1 <= rank <= 128, finite, and refusal of a whole-tensor rescale. `export_peft` writes a PEFT
adapter with `lora_A = A`, `lora_B = B`, `lora_alpha = r`, so the adapter's delta is exactly `B @ A`.
`controls.matched_control(edit, seed)` draws Gaussian `B', A'` per module and scales them to the same Frobenius norm
of `B A`.

## Cases and evaluation

- `target.Case`: prompt text or chat messages (rendered by `prompts.render_chat` with the checkpoint's template),
  role, split, group, and one rule: regexes, a check function or a contrast pair. Modes: `judged`, `contrast`,
  `unchanged`.
- `evaluate.run`: for each arm, inside `backend.edits(arm)`, every case: generation with seeds from
  `seed_of(f"{seed}|{case.id}", draw)` (identical across arms) or the log-prob margin of the contrast pair.
- `evaluate.side_effects`: per arm, KL(base || arm), top-1 agreement and NLL change at every position of reference
  texts (built-in prose and code, or yours), in chunks of 64 positions.
- `evaluate.compare`: target (one-sided exact sign test over cases against base and each control, group-clustered
  bootstrap CI of the mean improvement; controls are also tested against base), keep (cases broken per draw;
  "unchanged" cases whose greedy text changed), side effects, verdict. Thresholds in `Criteria`.

## Probing and construction

- `probe`: `divergence` (first differing token of two continuations, tokenised with the prompt), `margins`,
  `collect_tail` (one layer's MLP tail over many sequences), `measure_logit_shift`, `readout` (cross-validated AUC).
- `construct`: `fit_gate` (weighted ridge over MLP activations), `logit_direction` (minimum-norm direction through
  post-FFN norm, final norm and the tied unembedding), `gate_direction_edit`, `predict_logit_shift` (exact
  recomputation of the layer's tail; the true effect for the last layer), `unit_gain`, `ablate_output` /
  `ablate_input`, `quiet_units`.

## Kaggle / CUDA

The torch backend runs on CUDA with SDPA attention for the text model. Generation is a per-token Python loop with a
dynamic cache (about 100 ms per token on an L4 against 50 ms on an M4 Max in the study's measurements), so batching
independent prompts is the obvious next step for GPU throughput; the interface does not need to change for that.
