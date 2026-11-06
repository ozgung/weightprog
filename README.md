# weightprog

Gradient-free **weight programming** of Gemma 4 by frontier-model proposers, with a **pre-registered verification
gate**. One backend interface over **MLX** (Apple silicon) and **PyTorch** (CUDA/CPU).

A proposer (Claude Code, Codex, or you) gets a budget of forward passes through a server, reads activations,
forms a hypothesis about a failure, and writes a low-rank edit `W' = W + B A` in closed form: no gradients, no
training. The gate then decides, on held-out repositories the proposer never saw, whether the edit changes the
targeted behaviour beyond random edits of the same size, without breaking what already worked.

```
proposer (sandboxed)            forward-pass server              gate (pre-registered)
  kernel: read W, save B, A  ->  gen / read requests, budget  ->  frozen edit vs base vs matched random edits
  notebook, scratch              base cache, expect_edit pin      sign tests, Holm, guards, bootstrap, BLOCK on gaps
```

Status: alpha (0.1.0). Initial target model: Gemma 4 E4B. Tested offline on tiny random Gemma 4 configs only; see
[docs/real-model-validation.md](docs/real-model-validation.md) for the real-model checks and how to run them.

## What is in the box

| Module | What it does |
|---|---|
| `weightprog.backends` | `get_backend("mlx" \| "torch")`: load, tokenizer, `edits()` context (fp32 add, exact restore), `generate` (prefill once, cache copied per draw; draw 0 greedy), multi-segment `decode_segments` (generate / teacher-force on one cache), native `logprobs`, `capture` (resid, resid_pre, resid_mid, mlp, mlp_out, pli, attention probabilities, logits), `read_weight` |
| `weightprog.edits` | `ModuleEdit` / `EditSet` (npz with `module, B, A, meta`), validation (allowed modules, shapes, rank ≤ 128, finite, no whole-tensor rescale), norms, `export_peft` (adapter whose delta equals `B @ A` exactly) |
| `weightprog.controls` | matched random controls: same modules, ranks and per-module Frobenius norm, seeded |
| `weightprog.replay` | call records, the Gemma 4 tool-call codec (`<\|"\|>` strings, sorted keys), frozen-prefix replay and full-call replay (regenerate several arguments in order) |
| `weightprog.scoring` | escape-artefact, exact-copy and compile metrics; pluggable replacement function (exact match built in; optional adapter for the competition harness's `swegemma.edit.apply_replacement`) |
| `weightprog.gate` | `GateSpec` + `analyze()`: paired task-macro sign tests vs base and controls, Holm, guards, fixed-sequence secondary, bootstrap CIs; completeness checks that BLOCK; the full-call compile-guard amendment; a pre-registration template |
| `weightprog.construct` | closed-form constructors: direction ablation, "gate + direction" down_proj edits (ridge gates, least-squares logit directions), and `predict_logit_shift` (exact tail recomputation or first order) |
| `weightprog.proposer` | the budget server (file queue), the proposer-side kernel, a brief template, packet builder, macOS Seatbelt launchers for Claude Code and Codex, transcript audit |
| `skills/` | Agent Skills: `weight-programming` (proposer playbook), `edit-gate` (pre-register and run the gate), `gemma4-internals` |

## Install

```bash
pip install -e ".[mlx]"                  # Apple silicon
pip install -e ".[torch]"                # CUDA or CPU (transformers >= 5.5 with Gemma 4)
pip install -e ".[mlx,torch,peft,dev]"   # everything, for the tests
```

The core needs only numpy and tokenizers: the kernel, edits, controls, gate and scoring run without a deep-learning
framework (the safetensors reader widens bf16 itself).

## Quickstart (tiny random model, seconds on a laptop)

```python
import numpy as np
from weightprog.testing import write_tiny_model
from weightprog.backends import get_backend
from weightprog.edits import EditSet
from weightprog.construct import TailState, fit_gate, gate_direction_edit, logit_direction, predict_logit_shift

d = write_tiny_model("/tmp/tiny_g4")                       # random Gemma 4 layout, char tokenizer
be = get_backend("mlx", d, dtype="float32", device="cpu")  # or get_backend("torch", d, dtype="float32", device="cpu")
ids = be.tokenizer.encode("def f(x):\n    return x + 1  # some code to read")
pos = list(range(5, len(ids)))

# read the last layer's MLP inputs, then build a gate + direction edit in closed form
L = be.spec.n_layers - 1
st = TailState.from_backend(be, ids, pos, L)
e, x = be.tokenizer.token_to_id("e"), be.tokenizer.token_to_id("x")
direction = logit_direction(st.path, {e: -1.0, x: +1.0})          # lower "e", raise "x"
gate = fit_gate(st.act, np.array([float(i % 2 == 0) for i in range(len(pos))]), lam=1e-3)
edit = gate_direction_edit(L, gate[None], direction[None], alpha=2.0)

# predict, then measure
pred = predict_logit_shift(st, edit, [e, x], "exact")
base = be.capture(ids, pos, what={"logits"})["logits"][:, [e, x]]
with be.edits(EditSet([edit])):                            # applied for the block, restored exactly after
    new = be.capture(ids, pos, what={"logits"})["logits"][:, [e, x]]
print(np.abs(pred - (new - base)).max())                   # float32 rounding
```

The same flow from the command line (synthetic edit_file calls from `weightprog.testing.synthetic`):

```bash
weightprog tiny /tmp/tiny_g4
weightprog replay frozen --backend mlx --model-dir /tmp/tiny_g4 --dtype float32 --calls CS/calls.jsonl --keys CS/usable_keys.txt --out G/gen_base.jsonl
weightprog controls G/cand G/cand_rand1 --seed 20261115
weightprog score frozen G/gen_base.jsonl --calls CS/calls.jsonl --files CS/files
weightprog gate --gate-dir G --keys CS/usable_keys.txt --calls CS/calls.jsonl --files CS/files --m 1 cand
weightprog export-peft G/cand/edit adapter/
```

`tests/test_cli.py` runs this pipeline end to end.

## Case study: escape artefacts in Gemma 4 E4B's edit calls

Gemma 4 E4B, as the model inside a coding agent, often writes a literal `\n` or `\"` inside an `edit_file`
old_string where the file has a real newline or quote (an *escape artefact*; 100 of 640 base generations on the
fresh gate set). Two frontier models, each in a fresh sandboxed context with 25 forward passes on training-repo data
only, wrote frozen low-rank edits. The registered round-2 gate ran on 320 calls x 2 draws from 101 tasks in 69
held-out repositories. Full write-up: [docs/case-study-escape-artefacts.md](docs/case-study-escape-artefacts.md).

| | Opus 5.5 (`opus_r2`) | Fable 5.1 (`fable_r2`) |
|---|---|---|
| Edit | 4 modules in layers 40-41 (ranks 1, 40, 40, 40) | rank 2 on layer 41 `mlp.down_proj` |
| Escape artefacts, base -> edit (of 640) | 100 -> 77 | 100 -> 47 |
| Tasks with fewer / more artefacts | 9 / 0 | 20 / 0 |
| Primary p (max over base and two matched controls) / Holm threshold | 0.0107 / 0.05: pass | 9.5e-7 / 0.025: pass |
| Matched random controls vs base (artefacts) | -0.2 and +0.1 points | 0 and 0 |
| Repo-clustered 95 % CI of the artefact effect (points) | -7.7 to -1.2 | -15.4 to -5.1 |
| Acceptance, task-macro vs base (guard >= -1 point) | +0.39: pass | -1.01: fail |
| Served-ok calls lost, draw 0 / 1 (limit 2 each) | 2 / 3: fail | 5 / 5: fail |
| Registered verdict | not passed | not passed |

**Honest limits.** Both edits change the targeted behaviour on unseen repositories beyond matched random edits of
the same size. Neither passed the full registered gate: both broke a few copies the base model already got right
(the served-ok guard), and Fable's edit also missed acceptance non-inferiority by 0.01 points. No edit has been
shown to raise SWE task scores; the registered plan sends only a gate-passing edit to an agent-level test, so none
went. The study also found that its own registered compile guard was confounded by the frozen-prefix replay (the
"compile trap"; amended before any candidate generation, see `weightprog.gate.fullcall`).

## Repository layout

```
src/weightprog/   library (backends, edits, controls, replay, scoring, gate, construct, proposer, cli, testing)
skills/           Agent Skills (SKILL.md with name/description frontmatter)
docs/             architecture, case study, real-model validation
scripts/          real-model validation scripts (not run in CI)
tests/            pytest, offline, tiny random models only
```

## Tests

```bash
pytest -q        # offline; tiny random Gemma 4 configs; both backends on CPU
WEIGHTPROG_GATE_DIR=.../gate_r2 WEIGHTPROG_REPLAY_DIR=.../replay_ood pytest tests/test_gate_regression.py tests/test_controls.py
```

The regression tests read the study's scored generation files (not distributed) and check that the library gate
reproduces the registered report `gate_r2_report.json` exactly and that the matched controls are bit-identical.

## Data, weights and licences

- Code: Apache-2.0 (LICENSE, NOTICE).
- No model weights, no edit files, no task data, prompts, file states or gold patches are in this repository.
  Edits and adapters computed from Gemma weights are model derivatives governed by the Gemma terms of use;
  publishing them is your decision.
- `swegemma` (the competition harness) is not vendored; `SwegemmaReplacer` imports it only if it is installed.

## Credit

Weight-space programming by language-model proposers is prior work: Takehi, Teoh, Zhang and Khattab,
"weight-space programming" (https://rikiyat.github.io/blog/weight-space-programming/). This library adds a
two-backend toolkit for Gemma 4, agent tool-call replay and scoring, and a pre-registered gate with matched random
controls on held-out repositories.
