# weightprog

Edit a **Gemma 4** model's weights directly, in closed form, to change one behaviour: no gradient training. Hand
this repository to your coding agent (Claude Code, Codex, ...) together with a description of what you want changed;
it gives the agent the instruments, the constructions and a workflow, and it tells you honestly whether the edit
worked. Initial target model: **Gemma 4 E4B**. MLX (Apple silicon) and PyTorch (CUDA/CPU) backends.

```
your cases (JSONL)  ->  census of the base model  ->  locate the decision token, read activations
     -> closed-form edit W' = W + B A (gate + direction, ablation)  ->  predict its effect for free
     -> eval: base vs edit vs matched random edits, keep cases, KL/NLL side effects  ->  export as a LoRA adapter
```

**For agents:** start with [AGENTS.md](AGENTS.md), then [skills/weight-editing](skills/weight-editing/SKILL.md) and the
recipes in [docs/recipes](docs/recipes/).

Status: alpha (0.2). The backends are checked against the study's real-model generations on E4B
([docs/real-model-validation.md](docs/real-model-validation.md)); everything else is tested offline on tiny random
Gemma 4 configs.

## Install

```bash
pip install -e ".[mlx]"                  # Apple silicon
pip install -e ".[torch]"                # CUDA or CPU (transformers >= 5.5 with Gemma 4)
pip install -e ".[mlx,torch,peft,dev]"   # everything, for the tests
```

## Quickstart (tiny random model, seconds on a laptop)

```python
import numpy as np
from weightprog.testing import write_tiny_model
from weightprog.evaluate import open_model, run, compare
from weightprog.target import Case
from weightprog.probe import divergence, collect_tail
from weightprog.construct import LogitPath, fit_gate, logit_direction, gate_direction_edit, predict_logit_shift
from weightprog.edits import EditSet
from weightprog.controls import matched_control

d = write_tiny_model("/tmp/tiny_g4")                       # random Gemma 4 layout, character tokenizer
be = open_model(d, backend="mlx", dtype="float32", device="cpu")
tok, L = be.tokenizer, be.spec.n_layers - 1

# where does "x = obj." choose between "m..." and "d..."?
dv = divergence(tok, "x = obj.", "model()", "dict()")
st = collect_tail(be, [(dv.prefix_ids, [len(dv.prefix_ids) - 1]), (tok.encode("def f(a):\n    return a"), [3, 6, 9])], L)
gate = fit_gate(st.act, np.r_[1.0, 0.0, 0.0, 0.0], lam=1e-3)                  # on at the decision, off elsewhere
direction = logit_direction(LogitPath.from_backend(be, L), {dv.good_id: +1.0, dv.bad_id: -1.0})
edit = gate_direction_edit(L, gate[None], direction[None], alpha=3.0)
print(predict_logit_shift(st, edit, [dv.good_id, dv.bad_id]).round(2))        # exact, no forward pass

cases = [Case(id="t", prompt="x = obj.", good=[r"^m"], bad=[r"^d"], max_new=4),
         Case(id="k", prompt="def g(a):\n    return", role="keep", max_new=4)]
arms = {"base": None, "edit": EditSet([edit]), "rand1": matched_control(EditSet([edit]), seed=1)}
rows = run(be, cases, arms, draws=2)
print("\n".join(compare(rows, arm="edit", controls=["rand1"])["summary"]))   # on a random model: shows the calls only
```

On a real model the same steps run on your own cases; `docs/recipes/context-token-choice.md` is a complete worked
version and `examples/spelling/cases.jsonl` its case file (the case format only: on E4B those prompts never elicit the word, see the example's README).

## Command line

```bash
weightprog census --model-dir M --cases cases.jsonl --split tune -v           # how the base model behaves
weightprog eval --model-dir M --cases cases.jsonl --split tune --edit edit/ --controls 2 --side-effects --report r.json
weightprog render --cases cases.jsonl --model-dir M --id case-7                # the exact prompt text
weightprog validate-edit edit/ --model-dir M
weightprog export-peft edit/ adapter/                                          # PEFT LoRA, delta == B @ A exactly
weightprog tiny /tmp/tiny                                                      # a tiny random checkpoint for dry runs
```

## What is in the box

| Module | What it does |
|---|---|
| `weightprog.backends` | `get_backend("mlx" \| "torch")`: load, tokenizer, `edits()` context (fp32 add, exact restore), `generate` (draw 0 greedy, then sampled at the served settings), multi-segment decoding, native `logprobs`, `capture` (resid, resid_pre, resid_mid, mlp, mlp_out, pli, attention probabilities, logits), `read_weight` |
| `weightprog.target` | cases: prompt or chat messages, role (target / keep), split, group, a rule (regexes, a check function, or a contrast pair) |
| `weightprog.evaluate` | `open_model`, `run` (all cases under all arms, paired seeds; `reuse_rows` for cached base rows), `side_effects` (KL and NLL on held-apart reference prose, code and chat), `compare` + `Criteria` (target vs base and every matched control with a group-bootstrap check, keep cases against the controls' noise floor, side effects; verdict PASS / FAIL / INCOMPLETE) |
| `weightprog.probe` | `divergence` (the deciding token), `decision_from_text` (the deciding token inside the model's own generations), `margins`, `collect_tail` (one layer's MLP inputs over many examples), `layer_sweep` and `readout` (cross-validated AUC: where is the choice readable?), `patch_effect` / `patch_sweep` (activation patching: where is it caused?), `measure_logit_shift` (any layer) |
| `weightprog.construct` | closed-form constructors: gate + direction edits (ridge gates with `cv_gate`, least-squares logit directions), the covariance-whitened key -> value update (`second_moment`, `kv_update`), unit patching (`quiet_units`, `unit_patch`), null-space restriction (`null_space_basis`, `project_edit`), residual-direction writes for any layer, direction ablation, and `predict_logit_shift` (exact for the last layer) |
| `weightprog.edits` | `ModuleEdit` / `EditSet` (npz with `module, B, A, meta`; `merge` sums two edits of one module), validation (allowed modules, shapes, rank <= 128, finite, no whole-tensor rescale), norms, `realised_fraction` (what survives the bfloat16 cast), `export_peft` |
| `weightprog.controls` | matched random edits: same modules, ranks and per-module Frobenius norm, seeded |
| `weightprog.prompts` | `render_chat`: the checkpoint's own chat template (fallback: Gemma 4's turn layout) |
| `skills/` | Agent Skills: `weight-editing` (the workflow), `gemma4-internals` (architecture facts that matter for edits) |

## Where it came from, and what it has and has not shown

The library was built for, and then generalised from, a study in which frontier models edited Gemma 4 E4B to stop
it writing escaped text into a coding agent's edit calls. Every edit changed the target on 69 held-out repositories
beyond random edits of the same size, and none passed the study's pre-registered gate: each also broke a few
outputs the base model had right. Gradient-trained LoRA adapters failed the same way. What carries over to your own
target is in [docs/recipes/lessons-escape-artefacts.md](docs/recipes/lessons-escape-artefacts.md). The study's
experiment kit (tool-call replay, the registered gate, the sandboxed forward-pass server) lives in the project
repository as `weightprog_study`, not here.

## Does it help an agent?

A first test, with one run per arm: too small to settle the question either way. The target, new to the library and
fixed in advance: in projects pinned to pydantic 1, stop E4B writing pydantic 2 names (`.model_json_schema()` for
`.schema()`). Claude Opus 5.5 had four hours with the library and four without it; Claude Fable 5.2 with it followed.
A judge fixed in advance, written by the library's author, scored the edits on held-out cases.

| 54 held-out cases | Opus, library | Opus, no library | Fable, library |
|---|---|---|---|
| Wrong-version answers ↓ (base 0.29) | **0.00** | 0.04 | **0.00** |
| Correct pydantic 1 answers ↑ (base 0.25) | **0.56** | 0.40 | 0.40 |
| Side effects, KL ↓ (random edits 0.0006 to 0.0011) | 0.00007 | 0.00066 | **0.00004** |
| Unrelated plain-code outputs changed, of 4 ↓ | **1** | 2 | **1** |
| Verdict | fail | fail | fail |
| Time to the first evaluated edit ↓ | 33 min | **28 min** | 43 min |

↓ lower is better, ↑ higher is better, best in bold. Rates are over 48 outputs (24 cases, 2 draws each).

- **May help: the edit itself.** Both library runs chose `construct.unit_patch` on layer 41 on their own and built
  the cleaner edits: no wrong-version answers left and about a tenth of the side effects or less. The run without the
  library fitted a ridge gate on layers 30 and 31.
- **Did not help here: speed and the verdict.** Without the library, Opus wrote its own harness in 18 minutes and
  evaluated a first edit soonest. All three edits passed on the tuning cases and failed the same held-out check
  (four unrelated prompts must keep their greedy output), as would three of the six random edits.
- **Not yet known.** One run per arm cannot separate these differences from chance. Repeated runs, and a weaker agent
  for which the tooling may matter more, are the next tests. All three agents also wrote a small process to keep the
  model loaded; weightprog has no such session yet.

## Related methods

weightprog packages closed-form edits that are known from the model-editing and interpretability literature, for an
agent to apply to its own target, and adds the evaluation around them:

- rank-one and mass-editing of MLP weights as key -> value memories with a covariance-whitened update (ROME: Meng et
  al., 2022; MEMIT: Meng et al., 2023) - `construct.kv_update`;
- null-space constrained edits that leave preserved inputs unchanged (AlphaEdit: Fang et al., 2025) -
  `construct.null_space_basis`, `project_edit`;
- neuron patches (Knowledge Neurons: Dai et al., 2022; T-Patcher: Huang et al., 2023), here by rewiring existing
  low-use units in closed form - `construct.unit_patch`;
- activation patching / causal tracing to find where a behaviour is caused (Vig et al., 2020; Meng et al., 2022) -
  `probe.patch_effect`, `patch_sweep`;
- removing a feature by projecting a direction out of the weights (directional ablation, e.g. Arditi et al., 2024)
  - `construct.ablate_output` / `ablate_input`;
- adding a direction to the residual stream (activation steering: Turner et al., 2023; representation engineering:
  Zou et al., 2023), here written into the weights with Gemma 4's post-norms - `construct.residual_write_direction`;
- gated logit-space directions solved through the unembedding (logit lens: nostalgebraist, 2020), with an exact
  effect prediction for the last layer - `construct.gate_direction_edit`, `predict_logit_shift`;
- language-model proposers that write weights directly (weight-space programming: Takehi et al., see Credit).

## Tests

```bash
pytest -q        # offline; tiny random Gemma 4 configs; both backends on CPU; the recipes and the quickstart run
```

## Data, weights and licences

- Code: Apache-2.0 (LICENSE, NOTICE). The built-in reference texts for the side-effect check were written for this
  library and are under the same licence.
- No model weights, edits or adapters are in this repository. Edits and adapters computed from Gemma weights are
  model derivatives governed by the Gemma terms of use; publishing them is your decision.

## Credit

Weight-space programming by language-model proposers is prior work: Takehi, Teoh, Zhang and Khattab,
"weight-space programming" (https://rikiyat.github.io/blog/weight-space-programming/). weightprog adds a
two-backend toolkit for Gemma 4, case-based evaluation against matched random edits with keep cases and side
effects, and a workflow written for coding agents.
