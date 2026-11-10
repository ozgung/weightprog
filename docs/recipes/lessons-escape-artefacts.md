# Lessons from a study where the edits did not pass

weightprog grew out of a study on Gemma 4 E4B as the model inside a coding agent. The model often wrote a literal
`\n` or `\"` into an `edit_file` call's `old_string` where the file had a real newline or quote (an *escape
artefact*). Frontier models (Claude Opus 5.5, Claude Fable 5.1 and 5.2, GPT-6 Astra) were each given a fixed budget
of forward passes and asked to fix it with a closed-form weight edit. Every edit was judged by a pre-registered gate
on 320 calls from 69 held-out repositories (none in any editor's data), against matched random edits. The full study, its data and
the experiment kit are in the project repository (`weightprog_study/`); this page keeps what transfers.

## What happened

| Edit | Artefacts of 640 (base 100) | Beat base and random? | Correct copies broken per draw (limit 2) | Gate |
|---|---|---|---|---|
| Opus 5.5, 4 modules in layers 40-41 | 77 | yes | 2 and 3 | fail |
| Fable 5.1, rank 2 on layer 41 down_proj | 47 | yes | 5 and 5 | fail |
| Opus 5.5 with the earlier runs' notes, 3 modules of layer 41 | 25 | yes | 11 and 11 (acceptance -4.2 points) | fail |
| Gradient-trained LoRA for comparison (three arms: Fable's single module or all 258 matrices; 39 or 411 labelled calls) | 4-23 | yes | 4-13 | fail |
| Fable 5.2 with the earlier runs' notes; Fable 5.2 repairing the trained LoRA | none submitted | - | - | the editors withdrew their own best candidates after measuring the damage on fresh calls |

- **The target moved, specifically.** Every edit cut the artefact rate on held-out repositories beyond matched random
  edits of the same size, which changed nothing.
- **Collateral damage decided every verdict.** Each edit also broke a few copies the base model had right. More
  aggressive edits removed more artefacts and broke more. Gradient training did not escape this either.
- **The editors' own data understated the damage.** One editor measured about no change in acceptance on its search
  data; the held-out gate showed -4.2 points. The two Fable 5.2 editors that measured on fresh calls before
  submitting found the damage themselves and submitted nothing.
- **The defect turned out to be mostly harmless in its harness.** The agent harness already repaired most escaped
  copies, so removing the artefact rarely helped and every side effect cost real edits. In whole-agent runs the edit
  cut artefact-bearing calls sharply (task-level 0.060 to 0.009) without raising the number of solved tasks.

## What transfers

1. **Write keep cases first** and make them hard: the same surface form where the old behaviour is right (here:
   real backslashes in regexes and string literals, `\n\n` in newly written code).
2. **Check that fixing the target pays.** Measure what the defect costs downstream before engineering a fix; a
   cosmetic defect cannot repay any collateral damage.
3. **Hold out by group** (repository, template) and expect the holdout to look worse than tune.
4. **Weighted negatives and narrow gates** beat higher strength. Every strength sweep in the study traded more target
   effect for more broken keep cases on the same rows.
5. **The decision can be invisible at the row you edit.** The artefact was inherited from the model's own earlier
   text (P(artefact | escaped new_string) 73 % vs 3.7 % otherwise), and whether the earlier text was escaped was not
   linearly readable at the decision row. Read before you build.
6. **Judge on the same instrument as the user.** A replay that froze part of the model's output made one guard
   misleading (it paired an edited old_string with an unedited new_string); regenerating the whole call fixed it.
