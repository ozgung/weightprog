# Does it help an agent? Usability tests

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
