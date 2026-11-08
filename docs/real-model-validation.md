# Real-model validation

Everything in `tests/` runs on tiny random Gemma 4 checkpoints. The checks below need the real Gemma 4 E4B. They were
run once for this release (2026-11-07; results at the end) and all five passed. Run them again before using a backend
for a comparison that matters, after any mlx, mlx-lm, torch or transformers upgrade, and record the outputs next to the
study they serve.

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
agrees with the measured logit change to bf16 rounding on the top-20 base tokens; the linear prediction correlates
strongly with it. E4B's top logits lie mostly between 16 and 32 (the final soft cap is 30), where bf16 values are 0.125
apart, so a change measured as the difference of two bf16 forwards carries about 0.1-0.16 logit of rounding. (An
earlier version of this page said "a few hundredths"; that was too tight.) The first-order prediction is good for small
shifts but can miss large ones badly; use the exact prediction for strong edits. This also checks the per-layer-input
branch and layer scalar on the real weights.

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

## Results, 2026-11-07

weightprog 91edc21. MLX: Apple M4 Max, mlx 0.32.4, mlx-lm 0.32.2, bf16. Torch: NVIDIA L4 (Colab), torch 2.14.1+cu130,
transformers 5.22.1, bf16. Calls: the study's round-2 search set. Edit for check 3: the study's frozen Fable round-2
edit (layer 41 `down_proj`, rank 2).

| Check | MLX | Torch | Verdict |
|---|---|---|---|
| 1. MLX vs the study server's cache (40 calls) | greedy 40/40 and sampled 40/40 token-identical; full-call greedy 30/30 | n/a | pass |
| 2. Torch vs MLX (40 calls) | greedy text identical 40/40; full-call 39/40; log-probs of the recorded `old_string` median 0.000, max 0.48 nats | (same comparison) | pass; the max is one token at log-prob −3.8 against −3.3, and the next largest is 0.10 |
| 3. Tail prediction (8 calls) | exact error median 0.12, max 0.14 logit; linear correlation median 0.96 (min 0.92); measured shifts up to 13.8 | exact error median 0.12, max 0.16; correlation median 0.96 (min 0.92); shifts up to 13.9 | pass, against the corrected expectation above; on the 13.8 shift the linear prediction was off by 11.4 |
| 4. Capture vs native (3 longest prompts given to each run: MLX 17.8k-19.9k tokens, torch 15.8k-17.5k) | target tokens max 3.7e-4 nats; top-1 83/83; whole vocabulary row max 0.60 | target tokens max 1.1e-4; top-1 124/124; row max 0.86 | pass: differences sit in the deep tail of the row, as the study's server saw |
| 5. Gate reproduction | 5 tests passed | n/a | pass |

The raw outputs of these runs stay with the study; they contain competition task text and are not in this repository.

## What is untested on the real model

- MPS (torch on Apple GPUs): tested on CPU only.
- Generation with an edit applied, on either backend: check 3 applies an edit through `capture` only, and checks 1-2
  generate without an edit. (The study's own MLX server did generate with edits; that is not this code.)
- The budget server end to end with a real model and a real proposer: tested with the tiny model and synthetic
  calls (queue order, budget, refusals, pinning, base cache, full-call option, read arrays).
- The sandbox launchers and Seatbelt profiles: shell syntax checked, not launched in this repository (they are
  generalised from the study's launchers, which ran three proposers).
- PEFT export loaded into vLLM: tested with PEFT merge on the tiny HF model only.
