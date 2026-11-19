# weightprog

> **Version described in the Kaggle writeups (12 November 2026):** [v0.2.1](https://github.com/ozgung/weightprog/tree/v0.2.1) ([release](https://github.com/ozgung/weightprog/releases/tag/v0.2.1)). Later versions on `main` add changes made after the competition deadline: 0.2.2 (18 November 2026) ports fixes written by a GPT-6.2 Sol agent during a follow-up usability run; 0.2.3 (19 November 2026) adds `construct.residual_write_value`, post-norm-aware values for `kv_update` at any layer.

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
([docs/real-model-validation.md](docs/real-model-validation.md)). On real E4B, the usability test
[below](#does-it-help-an-agent) used the evaluation, the probes and several constructors (among them
`construct.unit_patch`), and adapters written by `edits.export_peft` were served by vLLM in the study. The rest is
tested offline on tiny random Gemma 4 configs.

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
| `weightprog.evaluate` | `open_model`, `run` (all cases under all arms, paired seeds; `reuse_rows` for cached base rows; `include_token_ids` keeps the generated token ids), `side_effects` (KL and NLL on held-apart reference prose, code and chat), `compare` + `Criteria` (target vs base and every matched control with a group-bootstrap check, keep cases against absolute limits (cases broken per draw, share of unchanged cases whose greedy text changed; the controls' figures are reported beside them), side effects; verdict PASS / FAIL / INCOMPLETE) |
| `weightprog.probe` | `divergence` (the deciding token), `decision_from_text` (the deciding token inside the model's own generations; `generated_ids=` follows the generated tokens exactly), `margins`, `collect_tail` (one layer's MLP inputs over many examples), `layer_sweep` and `readout` (cross-validated AUC: where is the choice readable?), `patch_effect` / `patch_sweep` (activation patching: where is it caused?), `measure_logit_shift` (any layer) |
| `weightprog.construct` | closed-form constructors: gate + direction edits (ridge gates with `cv_gate`, which can hold out whole groups of related prompts, least-squares logit directions), the covariance-whitened key -> value update (`second_moment`, `kv_update`; `residual_write_value` turns a wanted residual change into its values through Gemma 4's post-norm), unit patching (`quiet_units`, `unit_patch`), null-space restriction (`null_space_basis`, `project_edit`), residual-direction writes for any layer, direction ablation, and `predict_logit_shift` (exact for the last layer) |
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

Two small tests, with one run per arm: too small to settle the question either way. The target, new to the library
and fixed in advance: in projects pinned to pydantic 1, stop E4B writing pydantic 2 names (`.model_json_schema()` for
`.schema()`). Each agent had four hours. Claude Opus 5.5 ran once with the library and once without it, and Claude
Fable 5.2 ran with it (10-11 November 2026). GPT-6.2 Sol then ran once with the library, which it was allowed to
extend, and once without it (18 November). The same judge, fixed in advance and written by the library's author, scored
every edit on held-out cases.

| 54 held-out cases | Opus, library | Opus, no library | Fable, library | Sol, library | Sol, no library |
|---|---|---|---|---|---|
| Wrong-version answers ↓ (base 0.29) | **0.00** | 0.04 | **0.00** | **0.00** | **0.00** |
| Correct pydantic 1 answers ↑ (base 0.25) | **0.56** | 0.40 | 0.40 | **0.56** | **0.56** |
| Side effects, KL ↓ (random edits 0.0006 to 0.0011) | 0.00007 | 0.00066 | **0.00004** | 0.00023 | 0.00019 |
| Unrelated plain-code outputs changed, of 4 ↓ | 1 | 2 | 1 | **0** | **0** |
| Of 48 plain prompts, changed ↓ (random edits of the same size, median) | **3** (8) | 11 (12) | **3** (11) | 4 (12.5) | 5 (13) |
| Verdict | fail | fail | fail | **pass** | **pass** |
| Edit | units, layer 41 | ridge gate, layers 30-31 | units, layer 41 | units, layer 41 | units, layer 41 |
| Time to the first evaluated edit ↓ | 33 min | **28 min** | 43 min | not measured | not measured |

↓ lower is better, ↑ higher is better, best in bold. Rates are over 48 outputs (24 cases, 2 draws each). The 48-prompt
count was measured after the verdicts, with 20 random edits per arm. The Sol runs stopped on their own after 78 min
(library) and 109 min (no library).

- **May help: the edit itself, for Opus.** Opus with the library chose `construct.unit_patch` on layer 41 and built
  a cleaner edit than Opus without it (a ridge gate on layers 30 and 31): no wrong-version answers left, about a tenth
  of the side effects, and fewer plain prompts changed. Fable with the library did the same.
- **Made no difference for Sol.** With and without the library, Sol chose the same design (low-use units of layer
  41 rewired into detectors), and the two edits got identical target rates and both passed. Without the library it
  wrote its own tools (about 500 lines). With it, it fixed and extended the library; those changes are in 0.2.2.
- **Did not help here: speed.** Without the library, Opus wrote its own harness in 18 minutes and evaluated a first
  edit soonest.
- **The verdicts differ on a narrow check.** The three November 10-11 edits passed on the tuning cases and failed only
  the held-out check that four unrelated prompts keep their greedy output; the Sol edits changed none of the four.
  Random edits of the same size change none of the four about as often as not (25 of 60), so that check alone
  separates the edits weakly. On the 48-prompt count the Sol edits are no quieter than the Opus and Fable edits made
  with the library.
- **Not yet known.** One run per arm cannot separate these differences from chance. Repeated runs, and a weaker agent
  for which the tooling may matter more, are the next tests. Every agent also wrote a small process to keep the model
  loaded; weightprog has no such session yet.

## Methods

weightprog packages methods known from the model-editing and interpretability literature for an agent to apply to its
own target, and adds the evaluation around them. "On real E4B" says where a method has run on the real model, not
only on the tiny random Gemma 4 models of the tests; the rest is tested offline only.

| Method | In weightprog | From | On real E4B |
|---|---|---|---|
| **Find the behaviour** | | | |
| Activation capture at decision tokens | `backend.capture`, `probe.divergence`, `probe.decision_from_text`, `probe.collect_tail` | TransformerLens, nnsight, pyvene | checked against native log-probs ([validation](docs/real-model-validation.md), check 4) |
| Linear probes and read-outs | `probe.readout`, `probe.layer_sweep`, `construct.cv_gate` | Alain and Bengio, [2016](https://arxiv.org/abs/1610.01644) | used by agents in the usability runs |
| Difference of means | `probe.layer_sweep` (norm of the difference), captures | Zou et al., [2023](https://arxiv.org/abs/2310.01405) | tiny models only (the layer-30 edit computed it in its own script) |
| Logit lens, direct logit attribution | `construct.logit_direction`, `LogitPath` | nostalgebraist, [2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens); Elhage et al., [2021](https://transformer-circuits.pub/2021/framework/index.html) | used by agents in the usability runs |
| Exact prediction of an edit's effect | `construct.predict_logit_shift`, `TailState` (last layer, free); `probe.measure_logit_shift` (any layer, two passes) | this library | checked ([validation](docs/real-model-validation.md), check 3) |
| Activation patching, causal tracing | `probe.patch_effect`, `probe.patch_sweep` | Vig et al., [2020](https://arxiv.org/abs/2004.12265); Meng et al., [2022](https://arxiv.org/abs/2202.05262) | used by one agent run (GPT-6.2 Sol) |
| Attention and head analysis | attention probabilities in `capture` (building blocks only) | Olsson et al., [2022](https://arxiv.org/abs/2209.11895) | tiny models only |
| Sparse autoencoders | not in the library | Bricken et al., [2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html) | |
| **Change it** | | | |
| Unit (neuron) patching: rewire low-use MLP units into detector-writers | `construct.quiet_units`, `mlp_input`, `unit_patch` | Dai et al., [2022](https://arxiv.org/abs/2104.08696); Huang et al., [2023](https://arxiv.org/abs/2301.09785) | the judged edits of the library runs [below](#does-it-help-an-agent) |
| Gated logit-direction edit | `construct.fit_gate`, `cv_gate`, `gate_direction_edit` | cf. Transformer-Patcher (Huang et al., [2023](https://arxiv.org/abs/2301.09785)) | tried by agents in the usability runs; no judged edit |
| Key -> value update with covariance whitening | `construct.second_moment`, `kv_update`, `residual_write_value` | ROME (Meng et al., [2022](https://arxiv.org/abs/2202.05262)); MEMIT (Meng et al., [2023](https://arxiv.org/abs/2210.07229)) | one judged edit at layer 30 (Nov 2026): it moved the target on held-out cases and failed the keep check ([validation](docs/real-model-validation.md)); also tried by an agent in the usability runs |
| Steering direction written into the weights | `construct.residual_write_direction` (direction), `residual_write_value` (values for a kv update) | Turner et al., [2023](https://arxiv.org/abs/2308.10248); Zou et al., [2023](https://arxiv.org/abs/2310.01405) | the `residual_write_value` solve (then in the experiment's own script, same values) built that layer-30 edit; measured write cosine 0.96-0.99 at layers 30-38 for writes of the natural size, 0.90-0.96 at twice that |
| Null-space restriction | `construct.null_space_basis`, `project_edit`, `kv_update(..., null_basis=U)` | AlphaEdit (Fang et al., [2025](https://arxiv.org/abs/2410.02355)) | tiny models only |
| Directional ablation | `construct.ablate_output`, `ablate_input` | Arditi et al., [2024](https://arxiv.org/abs/2406.11717) | tiny models only |
| Attention-side edits | `gate_direction_edit(..., proj="o_proj")`, ablation of `o_proj` (no query, key or value edits) | | tiny models only |
| Scaling and merging edits | `ModuleEdit.scaled`, `EditSet.scaled`, `EditSet.merge` | task arithmetic (Ilharco et al., [2023](https://arxiv.org/abs/2212.04089)) | tiny models only |
| **Check and ship it** | | | |
| Matched random-edit controls, keep cases, KL and NLL side effects | `controls.matched_control`, `evaluate.run`, `side_effects`, `compare` | this library | every judged edit above |
| Export as a LoRA adapter | `edits.export_peft` | LoRA (Hu et al., [2022](https://arxiv.org/abs/2106.09685)) | adapters served by vLLM in the study |

Weight-space programming by language-model proposers, the idea of having a language model write weights directly, is
prior work (Takehi et al., see [Credit](#credit)).

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
  (Kaggle writeup, 2026): the library's design, method coverage and validation, with 186 offline held-out agent
  tasks and a 320-call replay set.
- *[Frontier Models as Weight Editors for Gemma 4 E4B](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/writeups/frontier-models-as-weight-editors-for-gemma-4-e4b)*
  (Kaggle writeup, 2026): the study the library came from.
