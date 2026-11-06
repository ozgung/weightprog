# Real-model validation

Everything in `tests/` runs on tiny random Gemma 4 checkpoints. The checks below need the real Gemma 4 E4B and have
**not been run** for this release. Run them before using a backend for a comparison that matters, and record the
outputs next to the study they serve.

Requirements: the E4B checkpoint directory (`config.json`, `model.safetensors` ~16 GB bf16, `tokenizer.json`); for
MLX a Mac with about 24 GB of free unified memory and no other model process on the GPU; for torch a CUDA GPU with
about 18 GB (the study's port peaked at 17.4 GB on an L4).

## 1. MLX backend vs the study server's cached base generations

```bash
python scripts/validate_mlx_vs_cache.py \
  --model-dir .../gemma-4-e4b-it-qat-q4_0-unquantized \
  --calls     .../outputs/weight_edit/shared/search_r2/calls.jsonl \
  --cache     .../outputs/weight_edit/wsp_cache/base_gen.jsonl \
  --fullcall-cache .../outputs/weight_edit/wsp_cache/base_fullcall.jsonl \
  --n 40 --out mlx_vs_cache.json
```

Pass: greedy (draw 0) and sampled (draw 1) generations token-identical in all or nearly all of the 40 calls; full-call
greedy pairs identical. Why it should pass: the backend's decode loop equals `mlx_lm.generate.generate_step` for
greedy and seeded sampled draws (tested on the tiny model), prefill uses the same 2048-token chunks, seeds are the
server's `seed_of(key, draw)`, and bf16 kernels are deterministic on one machine. Different mlx / mlx-lm versions
can change kernels; record the versions.

## 2. PyTorch backend vs MLX backend

```bash
python scripts/validate_torch_vs_mlx.py run --backend mlx   --model-dir M --calls C --keys K --n 40 --out mlx.jsonl
python scripts/validate_torch_vs_mlx.py run --backend torch --model-dir M --calls C --keys K --n 40 --out torch.jsonl --device cuda
python scripts/validate_torch_vs_mlx.py compare mlx.jsonl torch.jsonl
```

Pass: greedy frozen-prefix text identical in at least ~95 % of calls, outcomes identical after scoring, native
log-probs of the recorded old_string within a few tenths of a nat at worst and ~0 at the median. The study's earlier
port (same HF classes, an L4, 40 gate calls) measured greedy text identical 39/40, outcome 40/40, log-probs within
0.12 nats max, median 0.000. Sampled draws are not compared (different RNGs by design).

For a Kaggle notebook: install with `pip install -e ".[torch]"` (transformers with Gemma 4), attach the model,
and run step 2's torch half there; compare with an MLX file produced on a Mac.

## 3. Closed-form tail prediction on E4B

```bash
python scripts/validate_tail_prediction.py --backend mlx --model-dir M --calls C --edit RUN_OR_EDIT_DIR --n 8 --out tail.json
```

Needs any edit on `layers.41.mlp.down_proj` (for example a proposer's frozen edit). Pass: the exact tail prediction
agrees with the measured logit change to bf16 rounding (a few hundredths of a logit) on the top-20 base tokens; the
linear prediction correlates strongly with it. This also checks the per-layer-input branch and layer scalar on the
real weights.

## 4. Capture vs native forward on long prompts

`capture(..., what={"logits"})` against `logprobs` on a 14k-token prompt. In float32 they agree to 1e-4 on the tiny
model. In bf16 on E4B the study's server saw differences up to about 0.5 nats on long prompts between its manual
capture forward and the native cached forward (and at most 0.04 nats in a short smoke test), which is why read
requests report the native log-probs and keep the manual ones separately. Expect the same order here.

## 5. Gate reproduction (already automated)

```bash
WEIGHTPROG_GATE_DIR=.../outputs/weight_edit/claude/gate_r2 \
WEIGHTPROG_REPLAY_DIR=.../outputs/weight_edit/claude/replay_ood \
pytest tests/test_gate_regression.py tests/test_controls.py
```

This reads the study's scored generation files (read-only) and checks that `weightprog.gate.analyze` reproduces
`gate_r2_report.json` exactly, that the full-call summaries equal the recorded base / Opus numbers, and that the
stored matched controls are regenerated bit for bit. It passed for this release.

## What is untested on the real model

- Every backend operation on E4B weights (load, edits, generate, decode_segments, logprobs, capture): tested only on
  the tiny model, in float32 and (loading and loose agreement) bf16.
- Torch on CUDA and MPS: tested on CPU only.
- The budget server end to end with a real model and a real proposer: tested with the tiny model and synthetic
  calls (queue order, budget, refusals, pinning, base cache, full-call option, read arrays).
- The sandbox launchers and Seatbelt profiles: shell syntax checked, not launched in this repository (they are
  generalised from the study's launchers, which ran three proposers).
- PEFT export loaded into vLLM: tested with PEFT merge on the tiny HF model only.
