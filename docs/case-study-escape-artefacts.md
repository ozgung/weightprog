# Case study: programming out escape artefacts in Gemma 4 E4B, judged by a pre-registered gate

This is the study weightprog was extracted from (Kaggle "Gemma 4 developer agent" competition, November 2026). All
numbers are aggregates from the study's experiment record; dates refer to its dated Thread entries (UTC). No task
data, prompts, files or edits are reproduced here.

## The failure

Gemma 4 E4B (instruction-tuned QAT checkpoint, run in bf16) is the model inside a SWE-style coding agent. Its
`edit_file` tool takes `old_string` (text to replace, which must occur in the file) and `new_string`. About a third of
its edit calls fail with "old_string not found". One frequent, mechanical defect: the model writes a literal `\n`
(backslash, n) or `\"` (backslash, quote) inside old_string where the file has a real newline or quote. We call it an
**escape artefact**: the generated old_string contains `\n` or `\"` and the true file state does not. Text that
really contains a backslash (a `"\n"` literal in Python source) is not an artefact.

Base rates: 194 of 1,010 search-set generations and 42 of 450 on the round-1 selection set; on the round-2 fresh
gate set, 100 of 640 generations (30 of 101 tasks). The harness's unescape fallback already repairs many of them
(on the search set 110 of 194 artefact generations were accepted through it), so the artefact is partly cosmetic for
the harness; the rest fail as not found.

## Instruments

- **Replay**: recorded agent calls with their exact served prompt and the model's own prefix up to
  `old_string:<|"|>`; the arm regenerates old_string (greedy and one draw at T 0.6 / top_p 0.95 / top_k 64) and the
  result goes through the harness's edit function on the true file state. On the fresh set the base replay
  reproduces served behaviour: served-ok calls accepted 213/213 per draw, the recorded call replays to its recorded
  outcome 320/320.
- **Proposers**: frontier models in fresh contexts, each in a macOS sandbox with a read-only packet (brief,
  dossier of prior facts, search set of 730 calls from 118 tasks of the four training repositories: fastapi, rich,
  requests, httpx), a weight kernel (read tensors, save `W + BA`, rank <= 128, no whole-tensor rescale) and a budget
  of forward passes through the server (gen: regenerate up to 128 calls x 2 draws with base and/or edit; read:
  teacher-forced log-probs, residuals, MLP activations, attention). No gradients or training of any kind. Every
  transcript was audited (tools, network, paths, model loads).
- **Gate set**: 320 calls, 101 tasks, **69 repositories none of which any proposer saw** (two held-out pools,
  OOD-43 and OOD-B), x 2 draws.

## Round 1 (null)

Three frozen edits (two proposers at 10 passes, one recipe edit) on a 35-task selection set: task-macro effects
-0.0009 to -0.0022 on acceptance, p 0.75 to 0.88, matched random controls -0.0014 to +0.0046. No edit passed; one
also failed the served-ok guard. The pre-registered reading: no demonstrated benefit for these edits under this
budget and protocol (not equivalence to random, not proof that proposers cannot edit weights).

A separately registered rendering test (WP12) then showed that re-rendering tool responses in the chat template's
native form (file text raw instead of JSON-escaped) raises old_string acceptance at a fixed prefix: +0.0198
task-macro, 95 % CI [+0.0033, +0.0442], sign p 0.031, 5 tasks up and 0 down. Escape artefacts barely moved (42 -> 39),
pointing at the model's own earlier text (thought and new_string) as their source.

## Round 2 (registered 2026-11-04 17:1x, before any round-2 pass)

Target changed to the artefact itself. Proposers: Claude Opus 5.5 and Claude Fable 5.1 at effort max, 25 passes
each, at most 600 Python runs; a Codex GPT-6 Astra arm at 10 passes was registered separately (its own Holm family)
and ended with an empty edit, so it is not a candidate.

**Rule** (`GateSpec.round2()` reproduces it; the library regression test re-derives the registered report exactly):

1. Primary: per-task artefact rate; against base and both matched random controls (same modules, ranks and
   per-module Frobenius norm): a negative task-macro mean and a one-sided exact sign test p < 0.05; p = max of the
   three; Holm over m = 2.
2. Guards: task-macro acceptance >= base - 0.01; served-ok preservation (calls recorded ok that base accepts and the
   edit rejects) <= 2 per draw; compile-fail rate among accepted edits <= base + 2 points.
3. Key secondary (fixed sequence, only after a primary pass): acceptance against the same references.

### Results (registered verdict, analysed 2026-11-05 22:17, recorded 2026-11-06 09:33)

| | Opus (`opus_r2`) | Fable (`fable_r2`) |
|---|---|---|
| Escape artefacts, base -> edit (of 640) | 100 -> 77 | 100 -> 47 |
| Tasks with fewer / more artefacts | 9 / 0 | 20 / 0 |
| Primary p / Holm threshold | 0.0107 / 0.05: **pass** | 9.5e-7 / 0.025: **pass** |
| Random controls vs base (artefacts) | -0.2 and +0.1 points | 0 and 0 |
| Artefact effect, repo-clustered 95 % CI (points) | -7.7 to -1.2 | -15.4 to -5.1 |
| Acceptance (calls; task-macro vs base) | 443 -> 449; +0.39: pass | 443 -> 445; -1.01: **fail** |
| Served-ok lost, draw 0 / draw 1 (limit 2) | 2 / 3: **fail** | 5 / 5: **fail** |
| Compile guard, frozen-prefix metric | 9.0 % -> 12.5 %: fail | 9.0 % -> 15.3 %: fail |
| Registered primary / secondary | not passed / not tested | not passed / not tested |

Reported, not part of the rule: of the base model's 100 artefact generations Fable's edit leaves 47 and Opus's 77;
acceptance on those generations barely moves (61 -> 62 for Fable, 61 -> 63 for Opus) because the fallback already
accepted most. On recorded not-found calls Fable's edit raises acceptance from 15 to 27; on recorded-ok calls it
lowers it from 426 to 416. The edit trades calls: it rescues about a dozen failed copies and breaks about ten that
the base got right.

### The compile trap and the full-call amendment (2026-11-05 03:1x, before any candidate gate generation)

Gemma 4 writes tool arguments in sorted key order, so new_string comes before old_string. The frozen-prefix replay
therefore keeps the served (often escaped) new_string. When an edit turns an escaped old_string into an exact copy,
the harness stops taking its unescape fallback, which had also unescaped new_string, and inserts the frozen escaped
new_string verbatim: the file no longer compiles, whatever the edited model would have written. On the search set
every artefact-reducing pass of all three proposers raised the frozen-prefix compile-fail rate (+2 to +21 points;
in one pass 37 of 67 compile failures were exactly this case, against 2 of 31 for base). The guard of record became
the same +2-point rule on a **full-call replay** (both arguments generated by the arm; `weightprog.gate.fullcall`).
It cannot change the round-2 verdict, because the served-ok guard is unchanged and fails for both.

Descriptive full-call numbers for Opus (640 generations each, recorded 2026-11-06 14:07): accepted 444 -> 443;
compile-fail among accepted 7.9 % -> 8.4 % (+0.5 points, within the guard, where the frozen-prefix metric said +3.4);
old_string artefacts 89 -> 65; new_string artefacts 67 -> 47; exact copies 356 -> 381. The frozen-prefix compile
failure was a replay artefact for this edit. Fable's full call was still running when this was written.

## How the best edit works (Fable, rank 2 on layer 41 `mlp.down_proj`)

From the proposer's final answer, re-implemented generically in `weightprog.construct.logit_steer`:

- **Gates (rows of A)**: two closed-form weighted ridge regressions over 60 layer-41 GeGLU activations, fitted on
  teacher-forced rows, ~0.8-0.9 where the model is about to emit a punctuation+backslash token (`,\` `)\` ...) or an
  escaped-quote token (`\"` `=\"` ...), ~0 elsewhere and <= 0.35 where a legitimate backslash is copied. A third gate
  for bare `\` line ends fired on legitimate `"\n"` literals and was dropped.
- **Directions (columns of B)**: least squares through the post-FFN RMSNorm weight, the final RMSNorm weight and the
  tied unembedding, so that one unit lowers 49 escape tokens by one logit, raises the gate's raw family by 0.5,
  lowers newline tokens by 0.3 and holds raw counterparts and the 300 most frequent code tokens at 0 (spill-over on
  the other 261k tokens: median 0.08 logits per unit).
- **Strength**: 9.5 per unit gate; Frobenius norm 5.06 (10.5 % of the module's 48.3).
- **Calibration**: on 23 rows the observed change of the final residual had cosine 1.00 with the closed-form
  prediction and 0.48 of its norm: layer 41's layer scalar, which the first simulation had applied on the wrong side
  of a post-scalar residual capture (see the `weight-programming` skill).

Mechanism evidence the proposers reported (search data only): artefacts are confident, not marginal (escaped token
log-prob ~0, raw form ~10.6 nats lower at the median); they are inherited from the model's own text (73 % artefacts
when new_string was escaped vs 3.7 % otherwise); once a string turns raw it stays raw, so only the first escape
decision must be won; the decision forms late (escape evidence at global layers 35 and 41).

## What it shows, and what it does not

- Two frozen, gradient-free edits written by frontier models reduced the targeted failure on a fresh set of held-out
  repositories, beyond matched random edits of the same size (both primaries pass with the controls at zero).
- Neither did so without collateral loss on calls the base model already copied correctly; both fail the
  registered gate. No edit went to an agent-level test, and none has been shown to raise task resolution. Even a
  perfect copy-fidelity fix is bounded in end-to-end value: a failure census reported by a peer session (not
  re-verified in the record) put edit mechanics as the primary failure in about 12 % of E4B failures.
- A further registered arm (an "informed" Opus with access to the round-2 logs, 10 passes, full-call metric) was
  running when this was written; its result is not reported here.
