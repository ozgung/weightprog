# weightprog: start here (for coding agents)

You have been asked to change how a Gemma 4 model behaves by **editing its weights directly**: no gradient
training, no fine-tuning loop. weightprog gives you the instruments (load the model, read activations, measure
log-probabilities, generate), closed-form edit constructors, and an evaluation that tells you honestly whether your
edit worked. Initial target model: **Gemma 4 E4B** (instruction-tuned, 42 layers). Read this file fully, then
`skills/weight-editing/SKILL.md` (the workflow), then the recipe closest to your task in `docs/recipes/`.

## What an edit is

An edit is a set of low-rank additions to decoder-layer linear weights: `W' = W + B @ A` per module (q/k/v/o/gate/up/
down projections; rank at most 128). You compute `B` and `A` yourself, in closed form, from the weights and from
activations you capture: least squares, ridge regressions, projections. No optimiser, no loss minimised by
iteration. An edit is saved as a directory of `.npz` files and can be exported as a PEFT LoRA adapter, so anyone can
serve the edited model with standard tooling.

## The loop

1. **Write cases** (`weightprog.target`): JSONL, one prompt per line, with a rule that labels the model's output
   `good` / `bad` / `other` (regexes or a Python function), or a `contrast` pair of continuations. Mark the behaviour
   to change `role: target` and the behaviour that must survive `role: keep` (look-alikes and hard negatives above
   all). Split by `group` into `tune` (yours) and `holdout` (look once, at the end).
2. **Census** the unedited model on tune: `weightprog census ...` or `evaluate.run(be, cases)`. Confirm the defect
   exists and read the generations. Keep the base rows: you will reuse them in every evaluation.
3. **Find the decision**: `probe.divergence` gives the token where good and bad part ways; `probe.decision_from_text`
   finds it inside the model's own (census) generations, which is where the behaviour really happens;
   `probe.margins` tells how far apart the choices are (in nats). Capture activations there (`backend.capture`,
   `probe.collect_tail`) and check where the choice is readable (`probe.layer_sweep`, `probe.readout`) and where it is
   caused (`probe.patch_sweep`: activation patching between two prompts that differ only in the cause).
4. **Construct** an edit in closed form (`weightprog.construct`), on rows you fit and then check on other rows:
   - *gate + direction* on a late layer's `mlp.down_proj` (fires where the behaviour is about to happen, pushes the
     logits the right way); choose the ridge `lam` with `cv_gate`, not by in-sample gate values;
   - *covariance-whitened key -> value update* (`second_moment` + `kv_update`, the ROME/MEMIT closed form): hits the
     rows you give it and is quiet, by construction, in the directions ordinary text uses;
   - *residual direction* writes for a middle layer (`residual_write_direction`, for style or format behaviours with
     no single decision token), or *direction ablation* (`ablate_output`) to remove a feature;
   - *unit patching* (`quiet_units` + `unit_patch`): rewire a few low-use MLP units into a detector (gate row) that
     writes a chosen direction (down column);
   - any of these can be restricted to the *null space* of inputs you must preserve (`null_space_basis`,
     `project_edit`, `kv_update(..., null_basis=U)`, as in AlphaEdit): those inputs then see no change at all.
5. **Predict before you generate**: `construct.predict_logit_shift` (exact and free for the last layer's down_proj)
   or `probe.measure_logit_shift` (any layer, two forward passes) on target *and* keep rows; check
   `edits.realised_fraction` (a weak edit can be partly rounded away when cast back to bfloat16).
6. **Evaluate** on tune: `weightprog eval ... --base-rows base.jsonl` runs your edit and matched random controls
   (reusing the base rows), then checks target (beats base and every control, the good rate really rises), keep
   (nothing broken beyond the controls' noise) and side effects (KL and NLL on held-apart reference text). A quick
   look (`--controls 0`, no `--side-effects`) is fine while iterating, but its verdict is INCOMPLETE. Read the rows
   where something broke. Iterate.
7. **Finish**: evaluate once on holdout with criteria fixed in advance; save the edit and export it
   (`weightprog export-peft`).

## Ten lines of API

```python
from weightprog.evaluate import open_model, run, reuse_rows, side_effects, compare, Criteria, reference_texts
from weightprog.target import load_cases, select
from weightprog.probe import divergence, decision_from_text, margins, collect_tail, layer_sweep, patch_sweep, measure_logit_shift, readout, show_tokens
from weightprog.construct import LogitPath, TailState, fit_gate, cv_gate, logit_direction, gate_direction_edit, predict_logit_shift, unit_gain
from weightprog.construct import second_moment, kv_update, residual_write_direction, ablate_output
from weightprog.construct import quiet_units, mlp_input, unit_patch, null_space_basis, project_edit
from weightprog.edits import EditSet, ModuleEdit, realised_fraction
from weightprog.controls import matched_control

be = open_model(MODEL_DIR)                  # MLX on Apple silicon, PyTorch elsewhere; be.tokenizer, be.spec
with be.edits(EditSet([edit])):             # temporarily applied; restored bit for bit on exit
    lp = be.logprobs(ids, positions)        # [P, V] log-softmax; row j predicts token positions[j] + 1
cap = be.capture(ids, positions, layers=[40, 41], what={"resid", "mlp", "logits"})
outs = be.generate(ids, max_new=64, draws=2, seeds=[0, 1])   # draw 0 greedy, draw 1 sampled (T 0.6)
EditSet([edit]).save_dir("edit/")           # then: weightprog eval --edit edit/ ...
```

Full signatures: the docstrings of `weightprog/backends/base.py`, `evaluate.py`, `target.py`, `probe.py`,
`construct/logit_steer.py` and `construct/covariance.py`. Use `reference_texts("fit")` for gate negatives and leave
`reference_texts("eval")` (the default of `side_effects`) for the side-effect check. Architecture facts that matter for editing (layer types, KV sharing, norms, the layer
scalar, names): `skills/gemma4-internals/SKILL.md`.

## Rules that keep the result honest

- **Compare against matched random edits**, not only against the base model. `controls.matched_control(edit, seed)`
  has the same modules, ranks and per-module norm. If a random edit moves your metric, your effect is not specific.
- **Tune and holdout are split by group** (template family, source). `target.check_splits(cases)` lists leaks. Fix
  `Criteria` before the holdout run and run it once.
- **Keep cases are not optional.** Every edit that fires on the right rows also fires on some wrong ones. The study
  this library came from failed on exactly this: edits that clearly fixed the target broke a few outputs the base
  model got right (`docs/recipes/lessons-escape-artefacts.md`).
- **Judge generations, not only margins.** A flipped margin can still derail the text a few tokens later.
- **Report what failed.** An empty or partial result is a result.

## Compute

- E4B in bfloat16 needs about 16 GB of memory (MLX on a Mac with 32 GB or more; or a CUDA GPU with 24 GB). Loading
  takes tens of seconds; generation is about 50 ms per token on an M4 Max (measured on long prompts), one prompt at a
  time. Budget by tokens: 100 cases x 2 draws x 64 tokens = 12,800 tokens, about 11 minutes per arm, so a full `eval`
  with two controls (three generated arms with `--base-rows`, four without) takes 30-45 minutes. Give cases a `stop`
  (e.g. a closing code fence: generation ends at once) and short `max_new`; use `contrast` cases or `probe.margins`
  (one forward pass each) for the inner loop, and full generations to confirm.
- `second_moment` on E4B's MLP (10,240 units) holds an 840 MB matrix; a few thousand positions take minutes.
- Load the model once and keep it in one Python process for exploration; `be.edits(...)` switches edits in
  milliseconds. Only one process should hold the model on a Mac at a time.
- Tests and dry runs: `weightprog tiny /tmp/tiny` writes a random tiny Gemma 4 with a character tokenizer; every
  function works on it in seconds (`--dtype float32 --device cpu`).

## Where things are

```
src/weightprog/   backends (mlx, torch), edits, controls, construct, probe, target, evaluate, prompts, stats, cli
skills/           weight-editing (the workflow), gemma4-internals (architecture facts)
docs/recipes/     context-token-choice (gate + direction), remove-a-behaviour (direction ablation, edits outside the
                  last layer), global-preference (constant gate), lessons-escape-artefacts (a study where edits did
                  not pass, and what transfers)
tests/            offline tests on tiny random models (pytest -q)
```
