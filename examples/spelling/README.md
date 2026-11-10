# Example cases: `color` in US English, `colour` in British English

`cases.jsonl` is the case file of the worked recipe `docs/recipes/context-token-choice.md`: target cases ask for US
English (good: `color`), keep cases ask for British English (good: `colour`) or something unrelated (output must not
change). Tune and holdout use different subjects (`group`).

```bash
weightprog census --model-dir M --cases examples/spelling/cases.jsonl --split tune -v
weightprog eval --model-dir M --cases examples/spelling/cases.jsonl --split tune --edit edit_color --controls 2 --side-effects -v
```
