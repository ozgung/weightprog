---
name: weight-programming
description: Playbook for a proposer that edits Gemma 4 weights without training, through a forward-pass budget server. Use when you are given a weightprog brief, a run folder with requests/ and a weight kernel, and must change a behaviour with a low-rank closed-form edit (W' = W + B A) under a fixed pass budget.
---

# Weight programming (proposer playbook)

You change a model's behaviour by writing `W' = W + B @ A` on a few decoder-layer linears, computed in closed form
from weights and captured activations. No gradients, no optimisers, no iterative fitting to a loss. You run the
model only through the server; every pass is budgeted. Facts below come from the weightprog case study (Gemma 4 E4B,
escape artefacts); treat them as priors to check, not truths.

## Workflow

1. **Read the brief, dossier and server_spec first.** Write a notebook entry before every code run or request
   (what and why). Keep in it: hypotheses, evidence, current edit (with its `edit/` snapshot), next step.
2. **Look at the data before spending passes.** Census the search calls (outcomes, strata, where the defect occurs,
   what precedes it) with plain Python: it is free.
3. **Spend early passes on `read`, not `gen`.** Teacher-force recorded strings and forced alternatives; capture the
   residual (`resid`) and MLP activations (`mlp`) at the decision positions for a few late layers, attention for 1-2
   layers. Measure margins: how far is the wanted token from the chosen one? A defect at ~0 nats with the alternative
   10 nats lower (the study's escape artefacts) needs a decisive intervention at the decision row, not a nudge.
4. **Find where the decision is readable and where you can write.** A difference of means or ridge read-out on
   captured activations shows the layer where the coming choice is linearly separable. Late layers that reuse KV
   (E4B layers 24-41) change only the logits at the position being generated, never what later positions read: safe
   for output-local fixes.
5. **Construct in closed form** (`weightprog.construct`):
   - *Gate + direction* on `layers.L.mlp.down_proj`: rows of A are ridge gates over that layer's GeGLU activations
     (~1 where the defect is about to happen, ~0 elsewhere, explicitly ~0 on hard negatives such as legitimate
     backslashes); columns of B are least-squares directions through post-FFN norm -> final norm -> tied unembedding
     that lower the wrong tokens, raise their counterparts and hold frequent tokens at 0.
   - *Direction ablation*: `B = -V, A = V^T W` removes output directions V.
   - Before any gen pass, **predict** the logit shift on the rows you captured (`predict_logit_shift`, exact for the
     last layer) and on hard negatives. Only spend a gen pass on an edit whose prediction you believe.
6. **Gen passes compare base and edit on the same calls and seeds.** Base is cached and free after the first time.
   Pin the edit with `expect_edit`. Read the paired tables (only_base / only_edit), not just rates; open the rows
   where the edit lost an accepted call.
7. **Hold out within your search data.** Tune on one sample of tasks; confirm on a disjoint natural-proportion
   sample. The gate judges on repositories you never see; a fix tuned to one repository's code style will not carry.
8. **End cleanly**: what is in `edit/` when the budget is spent is your submission (empty is allowed); write
   `final_answer.md` with the edit, mechanism, evidence for and against, and the per-pass record.

## Pitfalls seen in the study

- **The layer-scalar factor.** Gemma 4 multiplies each layer's output by `layer_scalar` (0.48 on E4B layer 41). Residual
  captures are post-scalar; the MLP output is added pre-scalar. Adding a normed MLP change to a captured residual
  without multiplying by s overstated the effect by 1/s; the observed change was 0.48 x the predicted one with cosine
  1.00 (Fable, pass 6). `TailState` handles this; hand-written simulations must too.
- **The frozen-prefix compile trap.** Gemma writes tool arguments in sorted key order, so `new_string` precedes
  `old_string` and a frozen-prefix replay keeps the served new_string. Turning an escaped old_string into an exact
  copy makes the harness skip its unescape fallback and insert the still-escaped new_string verbatim, so the file
  stops compiling. Every artefact-reducing pass of all three study proposers raised the frozen-prefix compile-fail
  rate (+2 to +21 points). Judge compile effects on a full-call replay (`fullcall` in gen results when the run has it).
- **Collateral copy damage is the binding constraint.** Both round-2 edits beat base and matched random edits on the
  target and both failed the gate by breaking a few copies the base already got right (2-5 per draw against a limit
  of 2): dropped commas, merged-token jump-ahead, near-ties at the string close, mid-line quote flips. Make gates
  refuse those rows (weighted negatives) and inspect every lost call.
- **Context the edit cannot see.** Escapes were inherited from the model's own thought and new_string
  (P(artefact | escaped new_string) 73 % vs 3.7 %), and whether new_string was escaped was not linearly decodable at
  the decision row. Some cases are not fixable at the output; say so rather than over-fitting.
- **Legitimate look-alikes.** A gate for bare `\` line ends also fired on real `"\n"` literals and derailed them; it
  was dropped. Always include true positives of the surface form as negatives.
- **Strength sweeps are cheap to predict, expensive to run.** Fable's primary effect grew with strength (37 % ->
  50 % artefact reduction from alpha 6.5 to 9.5) while the lost copies stayed the same calls; decide the stopping
  rule before the sweep.
- **Random edits do nothing useful at the same norm** (a matched rank-2 random edit on the same module changed no
  generation). Beating base is therefore close to beating random for a targeted edit, but the gate tests both.

## Budgeting forward passes

In their first 10 passes the study's three blind proposers spent 3-6 passes reading internals; Fable used 6 of its
25 passes for reads and the rest for gen passes of up to 128 calls x 2 draws (a gen pass with an edit arm took
15-25 minutes on an M4 Max; with the full-call replay about twice that). A useful split:
2-4 reads to find the mechanism and calibrate, 1 gen pass per design change on the tuning sample, then 2-3 gen
passes on disjoint natural samples, and a final read under the submitted edit on hard negatives.
