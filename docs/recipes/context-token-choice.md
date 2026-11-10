# Recipe: make the model choose a different token in one context, and keep it elsewhere

Running example: when the user asks for **US English**, the model should write `color`; when they ask for
**British English** it should keep writing `colour`. The same shape fits many tasks: an API or keyword the model
picks wrongly in one setting and correctly in another, a name, a format choice. The construction is a rank-1
*gate + direction* edit on the last layer's `mlp.down_proj`:

    W' = W + alpha * d a^T

- `a` (the gate) reads the layer's MLP activations and is about 1 where the wrong choice is about to be made (US
  prompts, at the decision token) and about 0 everywhere else (British prompts at the same token, ordinary text);
- `d` (the direction) moves the logits: up for the wanted token, down for the unwanted one, zero for frequent tokens;
- `alpha` sets the push in nats.

Because E4B's layers 24-41 reuse earlier keys and values, a last-layer edit changes only the token being generated,
and `predict_logit_shift` computes its effect exactly from captured activations, without running the model.

## Code

```python
import numpy as np
from weightprog.evaluate import open_model, reference_texts
from weightprog.prompts import render_chat
from weightprog.probe import divergence, margins, collect_tail
from weightprog.construct import (LogitPath, fit_gate, cv_gate, logit_direction, gate_direction_edit,
                                  predict_logit_shift, unit_gain, second_moment, kv_update)
from weightprog.edits import EditSet, realised_fraction

MODEL_DIR = "/path/to/gemma-4-e4b-it"
be = open_model(MODEL_DIR)
tok = be.tokenizer
L = be.spec.n_layers - 1                      # the last layer: exact, free predictions

FIT = ["sky", "sea", "car", "dress", "logo", "wall", "flag", "sunset"]      # build the edit on these
CHECK = ["paint", "leaf", "bird", "coat"]                                   # look at these before generating

def text(dialect, subject):                   # the prompt plus the model's answer up to the word that decides
    msgs = [{"role": "user", "content": f"Write one sentence in {dialect} English about the {subject}'s shade."}]
    return render_chat(msgs, MODEL_DIR, prefill=f"The {subject}'s")

# 1. decision rows. divergence() tokenises both continuations with the prompt and splits at the first difference
def rows(subjects):
    us = [divergence(tok, text("US", s), " color", " colour") for s in subjects]        # should flip to " color"
    uk = [divergence(tok, text("British", s), " color", " colour") for s in subjects]   # must stay " colour"
    return us, uk
pos, neg = rows(FIT)
pos_c, neg_c = rows(CHECK)
good_id, bad_id = pos[0].good_id, pos[0].bad_id
trip = lambda ds: [(d.prefix_ids, d.good_id, d.bad_id) for d in ds]           # noqa: E731
m_pos, m_neg = margins(be, trip(pos)), margins(be, trip(neg))                 # log P(color) - log P(colour)
print("base margins: US", np.round(m_pos, 2), "British", np.round(m_neg, 2))

# 2. capture the last layer's MLP input at the decision rows and at ordinary rows (the "fit" reference texts;
#    the side-effect check uses the separate "eval" texts)
ordinary = [tok.encode(t) for t in reference_texts("fit")[:6]]
last = lambda ds: [(d.prefix_ids, [len(d.prefix_ids) - 1]) for d in ds]      # noqa: E731
items = last(pos) + last(neg) + [(ids, list(range(4, len(ids) - 1, 5))) for ids in ordinary]
st = collect_tail(be, items, L)
n_pos, n_hard = len(pos), len(neg)
n_ord = len(st.act) - n_pos - n_hard

# 3. gate: ~1 on US decision rows, ~0 on British ones (weighted up: false fires break keep cases) and ordinary rows;
#    choose the ridge strength out of fold, not from in-sample gate values
y = np.r_[np.ones(n_pos), np.zeros(n_hard + n_ord)]
w = np.r_[np.ones(n_pos), 4.0 * np.ones(n_hard), np.ones(n_ord)]
cv = cv_gate(st.act, y, weights=w, lams=(1.0, 10.0, 100.0))
lam = max(cv, key=lambda l: cv[l]["auc"])
print("cv:", {l: round(v["auc"], 3) for l, v in cv.items()}, "-> lam", lam)
gate = fit_gate(st.act, y, weights=w, lam=lam)

# 4. direction: +1 on " color", -1 on " colour", 0 on tokens the model often predicts on ordinary text
lp_ord = np.concatenate([be.logprobs(ids, p) for ids, p in items[n_pos + n_hard:]])
frequent = [int(t) for t in np.unique(lp_ord.argmax(-1)) if t not in (good_id, bad_id)][:200]
d = logit_direction(LogitPath.from_backend(be, L), {good_id: 1.0, bad_id: -1.0, **{t: 0.0 for t in frequent}})

# 5. strength: the margin moves by about 2 * alpha * gain * gate at a row (gain from unit_gain)
g = st.act @ gate
need = max(2.0, 2.0 - np.percentile(m_pos, 10))          # nats: lift the worst US rows to about +2
alpha = need / (2 * np.median(unit_gain(st)[:n_pos] * g[:n_pos]))
edit = gate_direction_edit(L, gate[None], d[None], alpha=alpha, note="color for US English")
print("realised after the bf16 cast:", realised_fraction(EditSet([edit]), be.read_weight))

# 6. predict (exact, free) on the CHECK rows, which the edit never saw, then measure on the real model
st_c = collect_tail(be, last(pos_c) + last(neg_c), L)
sh = predict_logit_shift(st_c, edit, [good_id, bad_id])
shift = sh[:, 0] - sh[:, 1]
print("check rows, predicted margin shift: US", np.round(shift[:len(pos_c)], 2), "British", np.round(shift[len(pos_c):], 2))
with be.edits(EditSet([edit])):
    m1_pos, m1_neg = margins(be, trip(pos_c)), margins(be, trip(neg_c))
print("check rows: US flipped", (m1_pos > 0).mean(), " British kept", (m1_neg < 0).mean())
EditSet([edit]).save_dir("edit_color")

# Variant: the covariance-whitened update (no gate to tune). It acts on the US decision rows' keys and stays small
# in the directions that ordinary text and the British rows use.
C, n = second_moment(be, [d_.prefix_ids for d_ in neg] + ordinary, L)
R = np.tile(alpha * np.median(g[:n_pos]) * d, (n_pos, 1))           # same push per row as the gated edit
kv = kv_update(L, st.act[:n_pos], R, C, lam=1.0, eps=1e-3 * np.trace(C) / len(C))
with be.edits(EditSet([kv])):
    print("kv variant on check rows: US flipped", (margins(be, trip(pos_c)) > 0).mean(),
          " British kept", (margins(be, trip(neg_c)) < 0).mean())
```

## Then evaluate on generations

Margins at one token are a proxy. Write cases (`cases.jsonl`) with held-out subjects and prompt wordings, then run
`weightprog eval`:

```json
{"id": "us-01", "messages": [{"role": "user", "content": "Describe a red barn in US English, one sentence."}], "role": "target", "split": "tune", "group": "barn", "good": ["\\bcolor"], "bad": ["\\bcolour"], "max_new": 48}
{"id": "uk-01", "messages": [{"role": "user", "content": "Describe a red barn in British English, one sentence."}], "role": "keep", "split": "tune", "group": "barn", "good": ["\\bcolour"], "bad": ["\\bcolor\\b"], "max_new": 48}
{"id": "other-01", "messages": [{"role": "user", "content": "List three prime numbers."}], "role": "keep", "split": "tune", "group": "misc", "max_new": 24}
```

```
weightprog eval --model-dir M --cases cases.jsonl --split tune --edit edit_color --controls 2 --side-effects -v
```

## When it does not work

- **The gate cannot separate the rows** (cross-validated AUC low, US and British check rows close): check
  `probe.layer_sweep`; try `support=` (only the units that differ most between the two groups), more varied
  positives (harvest them from generations with `probe.decision_from_text`), the covariance variant, or an earlier
  layer (`probe.measure_logit_shift` instead of `predict_logit_shift`; the change then also reaches later positions).
- **Keep cases break at near-ties**: lower `alpha`, weight hard negatives more, add the broken rows as negatives.
- **The flipped token derails the next ones**: look at the generations; the wanted continuation may need more than
  one token pushed (stack another rank-1 term for the second token, gated at its own decision row).
- **The behaviour is spread over many surface forms** (several tokenisations of the same choice): give the direction
  several targets (`{tok_a: +1, tok_b: +1, bad_a: -1, bad_b: -1}`) and the gate positives of every form.
