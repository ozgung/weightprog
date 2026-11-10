# Recipe: remove a behaviour by ablating a direction (and editing outside the last layer)

Running example: the model answers every question that mentions "steps" with a numbered Markdown list, even when a
sentence would do. Prompts that trigger the list and prompts that do not differ along a direction in the residual
stream; removing that direction from what the attention and MLP blocks write into the residual makes the behaviour
less likely. This is direction ablation as a
rank-1 edit per module:

    output side:  W' = (I - u u^T) W         B = -u,   A = u^T W

## Gemma 4 detail: ablate in the module's output space, not the residual's

Each block's output passes through a post-norm before it is added to the residual:
`residual += rms_norm(o_proj(...)) * w_post`. The contribution has no component along a residual direction `r` when
the module output is orthogonal to `w_post * r` (elementwise). So ablate `u = normalise(w_post * r)` from `o_proj`
(with `post_attention_layernorm.weight`) and from `down_proj` (with `post_feedforward_layernorm.weight`).

## Edits outside the last layer

- An edit in layer L < 24 changes the keys and values that later positions read, so its effect builds up over the
  whole continuation; layers 24-41 reuse the keys and values of layers 22/23 and change only the generated
  position.
- `predict_logit_shift` is exact only for the last layer. Elsewhere measure: `probe.measure_logit_shift` (two
  forward passes) and, in the end, generations through `weightprog eval`.

## Code

```python
import numpy as np
from weightprog.evaluate import open_model
from weightprog.prompts import render_chat
from weightprog.probe import measure_logit_shift, show_tokens
from weightprog.construct import ablate_output
from weightprog.edits import EditSet
from weightprog.names import module_name

MODEL_DIR = "/path/to/gemma-4-e4b-it"
be = open_model(MODEL_DIR)
tok = be.tokenizer
PRE = "model.language_model.layers"

with_b = ["What are the steps to boil an egg?", "Give me the steps to reset a router.", "What steps make a good tea?"]
without_b = ["How long should I boil an egg?", "Why does my router reset itself?", "What makes a good tea?"]
ids_w = [tok.encode(render_chat([{"role": "user", "content": q}], MODEL_DIR)) for q in with_b]
ids_o = [tok.encode(render_chat([{"role": "user", "content": q}], MODEL_DIR)) for q in without_b]

layers = list(range(max(1, be.spec.n_layers // 2 - 2), be.spec.n_layers // 2 + 2))   # a band of middle layers
mean_w = np.mean([be.capture(i, [len(i) - 1], layers=layers, what={"resid"})["resid"][:, 0] for i in ids_w], axis=0)
mean_o = np.mean([be.capture(i, [len(i) - 1], layers=layers, what={"resid"})["resid"][:, 0] for i in ids_o], axis=0)
r = (mean_w - mean_o).mean(0)                     # one residual direction, averaged over the band
r = r / np.linalg.norm(r)

edits = []
for l in layers:
    for proj, norm in (("o_proj", "post_attention_layernorm"), ("down_proj", "post_feedforward_layernorm")):
        name = module_name(l, proj)
        u = be.read_weight(f"{PRE}.{l}.{norm}.weight") * r
        edits.append(ablate_output(name, be.read_weight(name), u / np.linalg.norm(u)))
es = EditSet(edits)

list_start = [i for i, _ in show_tokens(tok, "1.")][:1]
for ids in ids_w:
    print("log P(list start) change:", measure_logit_shift(be, es, ids, [len(ids) - 1], list_start).round(3))
es.save_dir("edit_ablate")
```

Ablating the writes of a band of layers removes what those layers add along `r`, not what earlier layers already
wrote; if the direction is present from early on, ablate from the first layer where `probe.layer_sweep` finds it to
the last layer, or ablate `r` from the inputs of the layers that read it (`ablate_input`).

The direction from three pairs is noisy; use tens of pairs that differ only in the trigger, check that the direction
generalises with `probe.readout` on held-out pairs, and judge the edit on generations with keep cases (prompts where
the behaviour is appropriate, and unrelated prompts) and `--side-effects`.
