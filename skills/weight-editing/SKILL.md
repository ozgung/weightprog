---
name: weight-editing
description: Workflow for changing a Gemma 4 model's behaviour by a closed-form low-rank weight edit (W' = W + B A, no gradient training) with weightprog - define target and keep cases, census the base model, locate the decision token, read activations, construct a gate + direction or ablation edit, predict its effect, evaluate against matched random edits and side effects, export as a LoRA adapter. Use when asked to edit, steer or fix a Gemma 4 behaviour through its weights.
---

# Weight editing with weightprog (workflow)

The goal is an edit that changes one behaviour on prompts you have not tuned on, beats random edits of the same size,
leaves the cases you marked `keep` intact and barely moves the model on unrelated text. Most of the work is
measurement; the construction itself is a few lines of linear algebra.

## 0. Before anything

- Load the model once in a long-lived Python process (`be = open_model(model_dir)`) and keep a notebook file:
  what you ran, why, what came out. Save intermediate arrays (`np.save`) so a crash does not cost a re-run.
- Decide what success means now and write it into a `Criteria` (alpha, keep_max_broken, kl_max, nll_max). Do not
  change it after looking at holdout results.

## 1. Cases

Write 40-200 cases (`weightprog.target.Case`, JSONL). For each, the exact prompt the model will see (use `messages`
with the chat template for an instruction model, plus a short `prefill` such as an opening code fence so the
behaviour starts within a few tokens) and a rule:

- `good` / `bad` regexes on the generated text (a generation is `bad` if any bad pattern matches);
- or a `check` function for anything a regex cannot judge;
- or a `contrast` pair of continuations when the behaviour is one token choice (cheap: no generation).

Cover the behaviour in many surface forms (different names, styles, lengths, positions), and write `keep` cases:
the same surface form where the old behaviour is correct (hard negatives), near look-alikes, and unrelated prompts
(`keep` without a rule = output must not change). Give each case a `group` (its template or source) and split
groups into `tune` and `holdout`; never put one group in both (`check_splits`).

## 2. Census

`weightprog census --model-dir M --cases C --split tune -v` (or `evaluate.run`). Read the generations, not just the
rates. Typical findings that change the plan: the defect is rarer than you thought, the model does something
`other` (refuses, writes prose, uses a third form), or the keep cases already fail. Fix the cases and the rules
first; an edit cannot be judged by a broken ruler.

## 3. Locate the decision

For a target case, `probe.divergence(tok, prompt_text + text_up_to_the_choice, good, bad)` gives the shared prefix
and the first differing tokens; `probe.margins` gives log P(good) - log P(bad) there. Notes:

- Check the tokenisation (`probe.show_tokens`). The good token may be a common word piece that also starts other
  words; the bad token may be merged with punctuation. Pick the token that really carries the choice.
- The margin size tells you the strength you need: a defect at -3 nats needs a push of about +4 to +6 at the decision
  row, gated so it does not fire elsewhere.
- If generations go wrong *before* the token you picked, the decision is earlier; find it from the bad generations.
- **Harvest rows from the model's own generations**: `probe.decision_from_text(tok, prompt_text, generated, regex,
  group)` turns a census row (its generated text and a regex for the behaviour) into the prefix ids that end right
  before the deciding token. Positives = bad generations of target cases; hard negatives = good generations of keep
  cases. Hand-written prefills alone put the decision in a context the model may never produce.

## 4. Read

Capture at decision rows of target cases (positives) and at matching rows of keep cases and ordinary tokens
(negatives): `probe.collect_tail(be, [(ids, [pos]), ...], layer)` for the MLP input of one layer, or
`be.capture(..., what={"resid", "mlp"})` for several. `probe.readout(acts, labels)` gives a cross-validated AUC: where
it is near 1, a linear gate can separate the rows; `probe.layer_sweep(be, positives, negatives)` does it for every
layer at once and returns the difference of means per layer (a direction candidate). Readable is not the same as
used: `probe.patch_sweep(be, clean_ids, corrupt_ids, good_id, bad_id)` patches the residual of a prompt that shows
the behaviour with that of an otherwise identical prompt that does not, layer by layer, and reports how much of the
log-prob gap each layer restores (activation patching). On E4B, layers 24-41 reuse keys and values of layers 22/23, so an
edit there changes only the position being generated, never what later positions read: a safe place for
output-local fixes, and the last layer (41) has an exact, free effect predictor.

## 5. Construct (closed form)

Split your rows in two first: **fit** rows to build the edit, **check** rows (other cases, other groups) to look at
before any generation. In-sample gate values and predicted shifts look good almost by construction.

**Gate + direction on `layers.L.mlp.down_proj`** (recipe `docs/recipes/context-token-choice.md`):

```python
st = collect_tail(be, items, L)                         # rows: positives then negatives
y = np.r_[np.ones(n_pos), np.zeros(n_neg)]
w = np.r_[np.ones(n_pos), 3 * np.ones(n_neg)]           # weight negatives up: false fires are what break keep cases
cv = cv_gate(st.act, y, weights=w, lams=(1, 10, 100, 1000))   # out-of-fold AUC and gate levels per lam
lam = max(cv, key=lambda l: cv[l]["auc"])
gate = fit_gate(st.act, y, weights=w, lam=lam)          # a . act ~ 1 on positives, ~ 0 on negatives
path = LogitPath.from_backend(be, L)
d = logit_direction(path, {good_id: +1.0, bad_id: -1.0, **{t: 0.0 for t in frequent_ids}})
g = st.act @ gate
alpha = need / (2 * np.median(unit_gain(st)[:n_pos] * g[:n_pos]))   # margin moves by ~2 * alpha * gain * gate
edit = gate_direction_edit(L, gate[None], d[None], alpha=alpha)
```

**Covariance-whitened key -> value update** (`construct.second_moment` + `construct.kv_update`, the ROME/MEMIT
closed form): give it the activations at the rows where the edit must act (`K`, up to 128 rows) and what down_proj
should add there (`R`, e.g. `alpha * d` for each row), plus the second moment `C` of activations on ordinary text
(thousands of positions, e.g. `reference_texts("fit")` and your keep prompts). It hits those rows and stays small in
the directions ordinary text uses, with no gate to tune; `lam` trades the two. Try it when a gate leaks.

**Residual directions for any layer** (`construct.residual_write_direction`): turn a residual-stream direction (a
`layer_sweep` difference of means, a steering vector) into the module-output direction that writes it through Gemma
4's post-norm, then use it as the direction of a gate + direction edit (`gate_direction_edit(..., proj=...)`) or
of an ablation. For behaviours with no single decision token (style, format, tone).

**Unit patching** (`construct.quiet_units`, `mlp_input`, `unit_patch`): pick k low-use MLP units of a late layer and
rewrite them: gate row = a read-out of the condition (over `mlp_input` = the pre-FFN-normed residual), up row = a
read-out that is ~1 on ordinary rows, down column = the direction to write. Each unit is a small detector-writer; the
study's editors used this family most (40-80 units in layer 41).

**Keep inputs exactly unchanged** (`construct.null_space_basis` + `project_edit`, or `kv_update(..., null_basis=U)`;
AlphaEdit): from the second moment of activations at the keep rows, take the directions they do not use and restrict
the edit to them. The keep rows then see an exactly zero change at that module; the price is less room for the edit.

**Direction ablation** (`construct.ablate_output` / `ablate_input`): remove a feature direction from a module's
output or make it blind to an input direction (recipe `remove-a-behaviour.md`). **A constant push** everywhere: a gate
fitted to 1 on all rows (recipe `global-preference.md`). For several behaviours, stack rank-1 terms (one gate and
direction each) in one module: `EditSet.merge` adds two edits of the same module (`add` refuses a duplicate).

## 6. Predict, then measure cheaply

- `predict_logit_shift(st, edit, [good_id, bad_id])` on every captured row (exact for the last layer): positives
  should flip, negatives should not move. Count both.
- `probe.measure_logit_shift(be, EditSet([edit]), ids, positions, tokens)` for other layers or as a check.
- Margins under the edit on all tune cases (`with be.edits(...): margins(...)`) before any generation.
- Choose the strength from a small sweep of `alpha` on these predictions; decide the stopping rule before the sweep.
- `edits.realised_fraction(EditSet([edit]), be.read_weight)`: the share of the edit that survives the cast back to
  bfloat16. Well below 1 (weak, dense edits) means the model sees less than your prediction.

## 7. Evaluate on tune

```
weightprog eval --model-dir M --cases C --split tune --edit edit/ --controls 2 --side-effects \
    --criteria criteria.json --base-rows base_tune.jsonl --report tune.json -v
```

While iterating, `--controls 0` without `--side-effects` is a quick look (verdict INCOMPLETE); in Python, run base
once (`run(be, cases, {"base": None})`) and only the edit afterwards, then `compare(base_rows + edit_rows, ...)`.

Target must beat base and both controls (one-sided sign test over cases, group-bootstrap CI above 0) and the good
rate must rise (not just bad turning into `other`); keep must lose at most `keep_max_broken` cases per draw (the
controls' own breakage, printed beside it, is the noise floor of sampled draws); side effects must stay under the
limits (the controls' values show what "nothing specific" looks like). Open every broken keep case and every target case that got worse: is the gate firing on a
look-alike, is the push too strong at a near-tie, did the text derail after the flipped token? Then refit (more or
heavier negatives, a narrower `support` of units, lower alpha, another layer) and evaluate again.

## 8. Finish

Once, with criteria unchanged: `weightprog eval ... --split holdout --criteria criteria.json --report holdout.json`. Save `edit/`, the reports
and the notebook. `weightprog export-peft edit/ adapter/` writes a PEFT LoRA whose delta equals `B @ A` exactly
(serve it with vLLM, transformers or mlx_lm). Write down what the edit does, the evidence for and against, and what
you would try next.

## Pitfalls (from real runs on E4B)

- **Collateral damage is the binding constraint.** In the study behind this library, every edit that fixed its
  target on held-out repositories also broke a few outputs the base model had right, and none passed its keep guard.
  Weighted negatives, look-alike keep cases and reading every broken case are the main tools.
- **Your own tune data understates the damage.** One editor saw about no acceptance change on its own data and -4
  points on the held-out set. Keep holdout honest and expect it to be worse.
- **The layer scalar.** Gemma 4 multiplies each layer's output by `layer_scalar` (0.48 on E4B layer 41). Residual
  captures are post-scalar; the MLP output is added pre-scalar. `TailState` handles it; a hand simulation must too.
- **Near-ties.** Rows where good and bad are already close flip from tiny pushes; gates leak a little everywhere, so
  the keep cases at near-ties break first. Measure keep margins under the edit.
- **Prefix tokens.** Raising a token that is a prefix of several words can turn one wrong word into another wrong
  word. Look at the next few tokens of the generation, not only the flipped one.
- **Context the edit cannot see.** If the right choice depends on information that is not linearly readable at the
  decision row (readout AUC near 0.5), a gate cannot learn it; move earlier, add context to the cases, or say so.
- **Matched random edits do almost nothing** at the norms targeted edits use (a rank-2 random edit on the same module
  changed no generation in the study), so beating base usually means beating random; the test checks both anyway.
