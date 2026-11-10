# Example cases: `color` in US English, `colour` in British English

`cases.jsonl` is the case file of the worked recipe `docs/recipes/context-token-choice.md`: target cases ask for US
English (good: `color`), keep cases ask for British English (good: `colour`) or something unrelated (output must not
change). Tune and holdout use different subjects (`group`).

**Status (2026-11-10): the format, not a working target.** On Gemma 4 E4B these prompts do not elicit the word at
all: E4B names the shade instead ("a deep, rustic barn red"), so base, edit and random controls all label `other` and
no edit can be judged (`run_e4b.py` on E4B, and a census of a variant whose prompts contain the word). This is what
the census step of the workflow is for: run it before building anything, and change the cases (a prefill that forces
the word slot, or another target) when the base model never produces either label.

```bash
weightprog census --model-dir M --cases examples/spelling/cases.jsonl --split tune -v
weightprog eval --model-dir M --cases examples/spelling/cases.jsonl --split tune --edit edit_color --controls 2 --side-effects -v
```
