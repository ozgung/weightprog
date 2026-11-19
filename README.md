# weightprog

> The Kaggle writeups (12 November 2026) describe [v0.2.1](https://github.com/ozgung/weightprog/releases/tag/v0.2.1). Later versions add changes made after the competition deadline.

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

**For people:** paste this into Claude Code, Codex or a similar coding agent. It reads those instructions and asks
you for the rest (the behaviour to change, what must stay the same, where the model is):

```text
Help me change a behaviour of Gemma 4 E4B by editing its weights with weightprog
(https://github.com/ozgung/weightprog). Clone it, read AGENTS.md and follow it.
```

Status: alpha (0.2). Run on real Gemma 4 E4B: the backends (checked against the study's generations), the
evaluation, several probes and constructors, and LoRA export ([docs/real-model-validation.md](docs/real-model-validation.md)).
The rest is tested on tiny random Gemma 4 models.

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
| `weightprog.backends` | `get_backend("mlx" \| "torch")`: load, tokenizer, `edits()` context (exact restore), `generate`, native `logprobs`, `capture` (residuals, MLP activations and outputs, attention, logits), `read_weight` |
| `weightprog.target` | cases: prompt or chat messages, role (target / keep), split, group, a rule (regexes, a check function, or a contrast pair) |
| `weightprog.evaluate` | `open_model`, `run` (all cases under all arms, paired seeds), `side_effects` (KL and NLL on held-apart reference text), `compare` + `Criteria` (target vs base and every matched control, keep limits, side effects; verdict PASS / FAIL / INCOMPLETE) |
| `weightprog.probe` | `divergence` and `decision_from_text` (the deciding token), `margins`, `collect_tail`, `layer_sweep` and `readout` (where is the choice readable?), `patch_effect` / `patch_sweep` (where is it caused?), `measure_logit_shift` |
| `weightprog.construct` | closed-form constructors: gate + direction edits, key -> value updates (`kv_update`), unit patching, null-space restriction, residual-direction writes, direction ablation, and `predict_logit_shift` (exact for the last layer); see [Methods](#methods) |
| `weightprog.edits` | `ModuleEdit` / `EditSet` (npz with `module, B, A, meta`), validation (modules, shapes, rank <= 128), `realised_fraction` (what survives the bfloat16 cast), `export_peft` |
| `weightprog.controls` | matched random edits: same modules, ranks and per-module Frobenius norm, seeded |
| `weightprog.prompts` | `render_chat`: the checkpoint's own chat template (fallback: Gemma 4's turn layout) |
| `skills/` | Agent Skills: `weight-editing` (the workflow), `gemma4-internals` (architecture facts that matter for edits) |

## Other models

weightprog supports dense Gemma 4 models; only E4B is checked on the real model, and mixture-of-experts models are
not supported. The evaluation, the edit format and the linear algebra are model-independent; the backends and some
constructors are Gemma-specific. What another model would need: [docs/other-models.md](docs/other-models.md).

## Where it came from

The library was built for, and then generalised from, a study in which frontier models edited Gemma 4 E4B to stop
it writing escaped text into a coding agent's edit calls. Every edit changed the target on 69 held-out repositories
beyond random edits of the same size, and none passed the study's pre-registered gate: each also broke a few
outputs the base model had right. Gradient-trained LoRA adapters failed the same way. What carries over to your own
target is in [docs/recipes/lessons-escape-artefacts.md](docs/recipes/lessons-escape-artefacts.md). The study's
experiment kit (tool-call replay, the registered gate, the sandboxed forward-pass server) lives in the project
repository as `weightprog_study`, not here.

## Does it help an agent?

Two small tests on one new target (in projects pinned to pydantic 1, stop E4B writing pydantic 2 names), one run per
arm with four hours each, scored on 54 held-out cases by a judge fixed in advance:

| | Opus, library | Opus, no library | Fable, library | Sol, library | Sol, no library |
|---|---|---|---|---|---|
| Wrong-version answers (base 0.29) | 0.00 | 0.04 | 0.00 | 0.00 | 0.00 |
| Correct answers (base 0.25) | 0.56 | 0.40 | 0.40 | 0.56 | 0.56 |
| Verdict | fail | fail | fail | pass | pass |

With the library, Opus built a cleaner edit than without it; for GPT-6.2 Sol the library made no difference. The
three failures came from one narrow check (four unrelated prompts must keep their output), which random edits of the
same size also often fail. One run per arm cannot separate these differences from chance. Details, side effects and
timing: [docs/usability-tests.md](docs/usability-tests.md).

## Methods

Known methods from the model-editing and interpretability literature, packaged for an agent to apply to its own
target. References and validation status per method: [docs/methods.md](docs/methods.md).

| Method | In weightprog |
|---|---|
| Activation capture at decision tokens | `backend.capture`, `probe.divergence`, `probe.decision_from_text` |
| Linear probes, difference of means | `probe.readout`, `probe.layer_sweep`, `construct.cv_gate` |
| Logit lens, exact prediction of an edit's effect | `construct.logit_direction`, `predict_logit_shift`, `probe.measure_logit_shift` |
| Activation patching, causal tracing | `probe.patch_effect`, `probe.patch_sweep` |
| Unit (neuron) patching | `construct.quiet_units`, `unit_patch` |
| Gated logit-direction edit | `construct.fit_gate`, `gate_direction_edit` |
| Key -> value update (ROME, MEMIT) | `construct.second_moment`, `kv_update`, `residual_write_value` |
| Steering direction written into the weights | `construct.residual_write_direction`, `residual_write_value` |
| Null-space restriction (AlphaEdit) | `construct.null_space_basis`, `project_edit` |
| Directional ablation | `construct.ablate_output`, `ablate_input` |
| Matched random controls, keep cases, side effects | `controls.matched_control`, `evaluate.compare` |
| Export as a LoRA adapter | `edits.export_peft` |

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

## Citation

To cite the library itself:

```bibtex
@software{genc2026weightprog,
  author  = {Genc, Ozgun},
  title   = {{weightprog: closed-form weight editing for Gemma 4, with matched-control evaluation}},
  year    = {2026},
  version = {0.2.3},
  url     = {https://github.com/ozgung/weightprog},
  license = {Apache-2.0}
}
```

Please give the version you used; the Kaggle writeups describe 0.2.1. `CITATION.cff` holds the same reference for
GitHub's "Cite this repository".

For more background (not required for citing the library):

- *[Tools and Held-Out Tasks for Testing Agent-Written Weight Edits on Gemma 4](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/writeups/tools-held-out-tasks-agent-written-weight-edits)*
  (Kaggle writeup, 2026): the library's design and validation.
- *[Frontier Models as Weight Editors for Gemma 4 E4B](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/writeups/frontier-models-as-weight-editors-for-gemma-4-e4b)*
  (Kaggle writeup, 2026): the study the library came from.
