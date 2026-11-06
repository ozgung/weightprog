# Architecture

weightprog has three layers: **instruments** (backends, replay, scoring) that run the model the same way every
time, **artefacts** (edits, controls, call records, scored rows) that are plain files, and **judgement** (the gate)
that reads only artefacts. A proposer touches the instruments only through the budget server.

```
             +-------------------- proposer kit (weightprog.proposer) --------------------+
             |  packet (read-only): brief, dossier, server_spec, search/ calls + files,   |
             |  kernel.py (+ safetensors_np.py): read W, save edit/*.npz                  |
             |  run folder: requests/fwd_<k>.json -> .done.json, work/, edit/, notebook   |
             +------------------------------------|---------------------------------------+
                                                  v
   backends (mlx | torch)  <--  server: gen (replay + score), read (capture + logprobs), budget, pins, base cache
        |                                          
        v                                          
   replay.engine (frozen / full-call)  ->  scoring.EditScorer  ->  gen_<arm>.jsonl / .scored.jsonl
                                                                          |
   controls.matched_control (same modules, ranks, norms)                  v
                                                gate.io (BLOCK checks) -> gate.analyze (rule) -> report.json
                                                                       -> gate.fullcall.amend  -> amended.json
```

## Backends

`weightprog.backends.base.Backend` is the only interface the rest of the library uses.

| Method | Semantics |
|---|---|
| `load(model_dir, dtype, device)` | `config.json`, `model*.safetensors`, `tokenizer.json`. MLX: `mlx_lm.utils.load_model(strict=True)`, cast with `set_dtype`. Torch: `Gemma4ForConditionalGeneration` (or `Gemma4ForCausalLM` for text-only checkpoints), vision/audio towers dropped after load. |
| `tokenizer` | `tokenizers.Tokenizer` wrapper: no special tokens added, specials decoded verbatim (`<\|"\|>` is one token). Shared by both backends, so token ids are identical. |
| `spec` | `names.ModelSpec` from config: layer types, head dims, KV-sharing sources, softcap, editable modules. |
| `edits(edit_set)` | context manager: `W' = (W.float32 + B @ A).astype(W.dtype)`; originals kept and restored bit for bit on exit (also on error). Nesting works. |
| `generate(ids, max_new, draws, seeds, stop_ids, sampling)` | one Generate segment of `decode_segments`. |
| `decode_segments(ids, [Generate(max_new, stops) \| Force(ids)], draws, seeds)` | prefill `ids[:-1]` once (chunks of `prefill_chunk` = 2048), deep-copy the cache per draw, feed `ids[-1]`; draw 0 greedy, draws >= 1 sampled. A stop token ends a Generate segment and is not fed; a Force segment's tokens are fed before the next Generate. An unclosed Generate aborts the rest of the draw (`abort_on_unclosed`). |
| `logprobs(ids, positions)` | native cached forward (the generation code path), log-softmax over the vocabulary; row j predicts token `positions[j] + 1`. |
| `capture(ids, positions, layers, what, attn_layers)` | manual forward over the same modules. `resid` (after the layer, post layer-scalar), `resid_pre` (layer input), `resid_mid` (after the attention block: what the MLP output is added to), `mlp` (GeGLU activations entering down_proj), `mlp_out` (down_proj output, pre-norm), `pli` (per-layer input), `attn` (probabilities [heads, P, L] with the sliding window and KV sharing), `logits` (softcapped, float32). |
| `read_weight(name)` | float32 numpy copy of any parameter or buffer by canonical name. |

Canonical names are those of the HF Gemma4ForConditionalGeneration checkpoint
(`model.language_model.layers.{i}.{self_attn|mlp}.{proj}.weight`); MLX maps them to `language_model.model...`.

**Sampling.** Both backends implement mlx_lm's order: top-p on the untempered distribution, then top-k, then
categorical at temperature T (served settings T 0.6, top_p 0.95, top_k 64). The RNGs differ (MLX's global
`mx.random.seed(seed)` per draw; torch's `Generator(seed)` per draw), so **sampled draws are not token-identical
across backends**. Greedy agrees up to numerics (identical on the tiny test model in float32). Never mix generations
from different backends in one paired analysis: rows carry a `backend` field, and the server's base cache is per
backend.

**Seeds.** `seed_of(key, draw)` = first 8 hex digits of sha1("key|draw") (the forward-pass server's policy);
`index_seed(i, draw)` = 7919 i + draw (the study's stand-alone gate generators). Pick one per study and keep it.

**Parity tests** (`tests/test_parity.py`, `tests/test_backends.py`) build one tiny random checkpoint
(`weightprog.testing.tiny`: 6 layers, sliding window 8 with global layers 2 and 5, global head dim 32 with
proportional partial RoPE, layers 4-5 KV-shared without k/v projections, per-layer inputs, layer scalar 0.48 on
the last layer, softcap 6, tied embeddings) and check, in float32 on CPU: native log-probs agree to 1e-3; every
capture kind agrees to 2e-4 (relative to scale); greedy generations and multi-segment decodes are identical, also
under an edit; manual-forward logits equal native log-probs; attention rows sum to 1, are causal and zero outside
the window; the MLX decode loop equals `mlx_lm.generate.generate_step` for greedy and sampled draws.

## Edits and controls

An edit is a directory of `layers_{i}_{block}_{proj}.npz` files, each with `module` (canonical name), `B [out, r]`,
`A [r, in]` (float32) and `meta` (JSON string), exactly the format of the study's kernel. Validation: allowed module
(decoder-layer q/k/v/o/gate/up/down that exists in this model), shapes, 1 <= rank <= 128, finite, and refusal of a
whole-tensor rescale (`B A = c W`). `EditSet.snapshot(dir)` gives `{file: sha256}`, the value of the server's
`expect_edit` pin. `export_peft` writes an adapter with `lora_A = A`, `lora_B = B` and `lora_alpha = r` per module
(rank/alpha patterns when ranks differ), so the PEFT delta is exactly `B @ A` (tested by merging with PEFT).

`controls.matched_control(edit, seed)` draws Gaussian `B', A'` per module and scales both by
`sqrt(||BA||_F / ||B'A'||_F)`; module order and RNG use are those of the study's script, so the stored round-2
controls are reproduced bit for bit (regression test).

## Replay and scoring

A call record (`replay.records`) holds the exact prompt the served model saw, its own prefix up to the opening
delimiter of the argument to regenerate, the recorded arguments and outcome, and the file state's sha. The Gemma 4
tool-call format is `<|tool_call>call:NAME{k:v,...}<tool_call|>` with sorted keys and `<|"|>`-delimited strings
(`replay.codec`); sorted keys mean edit_file writes `new_string` before `old_string`, which is why the frozen-prefix
replay keeps the served new_string.

- **Frozen-prefix replay**: prompt + prefix, generate until `<|"|>`, cap min(1024, 2n + 64).
- **Full-call replay**: cut after `new_string:<|"|>`, generate new_string, force `<|"|>,old_string:<|"|>`, generate
  old_string. `regenerate_args` generalises this to any argument sequence.

`scoring.EditScorer` turns a generation into a row (outcome, accepted, exact_copy, compiles, escape_artefact, and
for full calls artefact_old/new and the frozen-new compile check). The replacement function is pluggable:
`ExactReplacer` (unique exact match) or `SwegemmaReplacer` (the competition harness, if installed). Results depend on
the replacer; the study's numbers come from the harness.

## The gate

`gate.analyze(base, candidates, controls, keys, cluster_of, spec)` implements the registered rule with every
threshold in `GateSpec` (defaults: primary escape_artefact lower-better, alpha 0.05, controls rand1/rand2, acceptance
non-inferiority margin 0.01, served-ok preservation max(2, floor(1 %)) per draw, compile margin 0.02, bootstrap
10,000 resamples with seed 20261104; `GateSpec.round2()` adds the study's reported sources). The unit is the task; the
cluster for the bootstrap is the repository. Statistics are exact and dependency-free (`weightprog.stats`).
`gate.io.load_arm` BLOCKs on duplicates, missing or extra records, an edit arm whose rows do not name its own frozen
edit directory, or a base row with an edit. `gate.fullcall.amend` replaces only the compile guard with the full-call
one and recomputes the fixed-sequence secondary over the same Holm family.

## The proposer kit

- **Server** (`proposer.server`, `weightprog serve --config server.json`): scans `RUNS_DIR/*/requests/fwd_*.json`,
  serves each run's requests in numeric order, claims by rename, writes `.done.json`, counts a pass only on
  success, refuses past the budget, checks `expect_edit`, caches base generations per (mode, rendering, backend),
  optionally adds a full-call replay to every gen request (`init-run --fullcall`).
- **Kernel** (`proposer.kernel`): numpy-only, copied into the packet with `safetensors_np.py` and a
  `kernel_config.json` naming the model directory, so a sandboxed proposer can read weights and save edits without
  importing the library.
- **Packet** (`weightprog packet`): brief, dossier, server spec, call set, kernel, optional `prior/`, writable
  `scratch/`; everything else read-only.
- **Sandbox** (macOS only): `launch_claude.sh` / `launch_codex.sh` run the proposer under `sandbox-exec` with
  `proposer_claude.sb` / `proposer_codex.sb`: home directory and shared temp unreadable except the packet, run
  folder, model, Python environment and the agent's own config; writes only to the run folder, scratch and a private
  temp dir; env cleared; no user settings, hooks, plugins or MCP servers. Known holes are listed in the profiles
  (metadata reads, open outbound TCP for the API); the transcript audit (`weightprog audit`) checks for network use,
  foreign paths, other tools and model loads.

## Kaggle / CUDA

The torch backend runs on CUDA with SDPA attention for the text model (eager for any towers before they are
dropped). Generation is a per-token Python loop with a dynamic cache; the study measured an earlier port at about
100 ms per token on an L4 against 50 ms on an M4 Max, so batching calls (independent prompts) is the obvious next
step for GPU throughput. The interface does not need to change for that.
