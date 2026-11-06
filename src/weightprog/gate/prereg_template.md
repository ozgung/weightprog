# Pre-registration: weight-edit gate for `{STUDY}`

Freeze this file (commit it and record its sha256) **before** any candidate edit is generated on the gate set.
After that, the only allowed changes are amendments written as dated entries below, each stating what was looked at
before it was made. Fill every `{...}` field; delete nothing.

## 1. Question

- Targeted behaviour (one sentence): {BEHAVIOUR, e.g. "literal backslash-n or backslash-quote in edit_file old_string where the file has none"}
- Base model and precision: {MODEL, e.g. Gemma 4 E4B QAT, bf16}; backend {mlx | torch} (generations of different backends are never mixed in one analysis)
- Candidates (the family): {CANDIDATE RUN IDS}; Holm family size m = {M} (a run whose final edit is empty is not a candidate and holds a p = 1 slot)

## 2. Gate set

- Source: {WHERE THE CALLS COME FROM}; held out from every proposer by {HOW: repositories, not just tasks}
- Universe: {N_CALLS} calls from {N_TASKS} tasks in {N_REPOS} repositories, listed in `{KEYS_FILE}` (sha256 {SHA})
- Draws per call: {DRAWS, e.g. 0 greedy, 1 sampled at T 0.6 / top_p 0.95 / top_k 64}; seeds {SEED POLICY}
- Replay: {frozen-prefix | full-call} for the primary; full-call for the compile guard (see "compile trap" in the edit-gate skill)
- Disclosed looks at the gate set before freezing: {e.g. "base arm only: 100/640 artefacts"}

## 3. References

- Base: the unedited model, same replay, same seeds (one base arm shared by all candidates)
- Matched random controls: {N_CONTROLS} per candidate, same modules, ranks and per-module Frobenius norm
  (`weightprog controls`), seeds `{SEED_BASE} + 10 i + k` (candidate i = 1.., control k = 1..)

## 4. Rule (GateSpec)

```json
{GATESPEC_JSON}
```

1. **Primary.** Per task, mean of `{PRIMARY_METRIC}` over its calls x draws. Against base and every control: a
   task-macro mean difference in the wanted direction AND a one-sided exact sign test p < {ALPHA}.
   p = max over the references; Holm over the family.
2. **Guards (all required).** Non-inferiority: task-macro `{NI_METRIC}` difference vs base >= -{NI_MARGIN}.
   Preservation, per draw: calls recorded `{PRESERVATION_KIND}` that base gets right and the edit gets wrong
   <= max({PRESERVATION_MIN}, floor({PRESERVATION_FRAC} x n)). Compile: fail rate among accepted edits <= base + {COMPILE_MARGIN}.
3. **Key secondary (fixed sequence).** Only for primary passes (otherwise p = 1): `{SECONDARY_METRIC}` against the same
   references, same test; Holm over the same family size.

Reported, not part of the rule: bootstrap CIs (tasks and repository clusters, {BOOT_N} resamples, seed {BOOT_SEED}),
per-source and per-recorded-kind tables, the fate of base-positive generations under the edit.

## 5. Completeness (BLOCK, no report)

Duplicate records; any arm missing or adding (key, draw) records in generation or scoring; an edit arm whose records
do not name its frozen edit directory; a base record with an edit; a candidate set that differs from section 1.

## 6. What each outcome means (written now)

- Primary and guards pass: "the edit changes the targeted behaviour on unseen {UNIT} beyond random edits of the same size, without the measured collateral damage."
- Primary passes, a guard fails: report both; the edit is not a pass.
- Primary fails: null for this candidate.
- In every case: no claim about downstream task success unless a separate, registered agent-level test shows it.

## 7. Frozen artefacts

| artefact | path | sha256 |
|---|---|---|
| this file | | |
| key file | | |
| candidate edits (frozen copies) | | |
| analysis code (weightprog version / commit) | | |

## Amendments

<!-- YYYY-MM-DD HH:MM UTC, who: what changed, why, what had been looked at before the change. -->
