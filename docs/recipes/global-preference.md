# Recipe: a small, global preference between tokens

Running example: the model overuses a word (`delve`); you want it to pick a plain alternative (`explore`) a little
more often, everywhere, without a context condition. The construction is a *constant gate*: a gate fitted to output
about 1 on every row, so the direction is added at every position with roughly the same strength.

Use this only for small pushes (a fraction of a nat to about 1 nat). A large constant push changes the model
everywhere the token could legitimately appear; for anything context-dependent use
`context-token-choice.md`.

## Code

```python
import numpy as np
from weightprog.evaluate import open_model, reference_texts
from weightprog.probe import collect_tail, show_tokens
from weightprog.construct import LogitPath, fit_gate, logit_direction, gate_direction_edit, predict_logit_shift, unit_gain
from weightprog.edits import EditSet

MODEL_DIR = "/path/to/gemma-4-e4b-it"
be = open_model(MODEL_DIR)
tok = be.tokenizer
L = be.spec.n_layers - 1

bad_ids = [i for i, _ in show_tokens(tok, " delve")][:1]       # check the pieces first: pick the deciding one
good_ids = [i for i, _ in show_tokens(tok, " explore")][:1]

# rows from varied text: the gate must be ~1 on all of them
texts = [tok.encode(t) for t in reference_texts()]
items = [(ids, list(range(2, len(ids) - 1, 3))) for ids in texts]
st = collect_tail(be, items, L)
gate = fit_gate(st.act, np.ones(len(st.act)), lam=1.0)
g = st.act @ gate
print("gate mean", g.mean().round(3), "sd", g.std().round(3))     # want mean ~1, small spread

d = logit_direction(LogitPath.from_backend(be, L), {**{t: 1.0 for t in good_ids}, **{t: -1.0 for t in bad_ids}})
push = 0.7                                                       # nats on the margin, small on purpose
alpha = push / (2 * np.median(unit_gain(st) * g))
edit = gate_direction_edit(L, gate[None], d[None], alpha=alpha, note="prefer explore over delve")
shift = predict_logit_shift(st, edit, good_ids + bad_ids)
print("predicted margin shift: median", np.median(shift[:, 0] - shift[:, 1]).round(3))
EditSet([edit]).save_dir("edit_pref")
```

Evaluate it with `weightprog eval --side-effects`: a global edit shows up in the KL on reference text, so compare it
with the matched random controls' KL and with your `kl_max`.
