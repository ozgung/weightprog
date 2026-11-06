# Task brief for a weight-editing proposer. Identical for every proposer except the values in this header.

You are the editor in a weight-editing experiment. Your run folder is `{RUN_DIR}`. Your forward-pass budget is
**{BUDGET} passes**. You are proposer `{PROPOSER}` ({MODEL}, effort {EFFORT}). The current directory is your
read-only packet: `brief.md` (this file), `dossier.md`, `server_spec.md`, `search/`, `kernel.py`, and a writable
`scratch/`.

## Goal

{MODEL_NAME} is the model inside {SYSTEM}. {BEHAVIOUR}

Your task: **change the model's weights, without any training, so that {TARGET}, without {GUARDRAIL}**, and
without making the model worse at anything else. {FAITHFULNESS} You choose what to look at, what hypothesis to test
and what edit to make.

Read `dossier.md` first: it records what is already known (earlier rounds, base rates, what did not work). Your edit
will be judged on held-out data you cannot see, by the rule at the end of the dossier: {JUDGING_SUMMARY}

## What you have

1. **Weights (read-only) and edit saving**, from Python started as `{PYTHON} -I`:
   ```python
   import sys; sys.path.insert(0, "{PACKET}"); import kernel as K
   K.names(regex); K.shape(name); W = K.load(name)            # float32 numpy, [out, in]
   E = K.rows("model.language_model.embed_tokens.weight", ids)  # selected rows only
   K.save_edit("{RUN_DIR}", name, B, A, note="...")         # W' = W + B @ A, B [out,r], A [r,in], r <= 128
   K.list_edits("{RUN_DIR}"); K.remove_edit("{RUN_DIR}", name)
   ```
   Editable: `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` of the decoder layers ({N_EDITABLE}
   matrices; KV-shared layers have no k/v projections, they reuse earlier layers' keys and values). Architecture:
   `{MODEL_DIR}/config.json`. {ARCH_NOTES} At most **{MAX_PY_RUNS} Python runs** for the whole run; keep scripts
   and intermediate arrays in `{RUN_DIR}/scratch/` or the packet's `scratch/`.
2. **Forward-pass server** (the only way to run the model). Write a request file
   `{RUN_DIR}/requests/fwd_<k>.json` (k = 1, 2, ...) and wait for `fwd_<k>.done.json`. {PASS_TIMING} Another
   proposer's requests may be queued first. {WAIT_NOTE} One request = one pass, whatever its size within the limits;
   a failed or refused request does not count. Full specification and limits: `server_spec.md`. In short:
   - `gen`: the model, base and/or with your current edit, regenerates the last argument at up to {GEN_CALLS}
     recorded calls (the exact prompt the agent saw plus the model's own text up to the argument's opening
     delimiter), 1 or 2 draws (greedy, then sampled); every result is scored against the true file state. You get
     acceptance, not-found, exact-copy and {METRIC_NAME} rates per arm, per recorded outcome, and paired
     base-vs-edit tables; per-call rows with the generated text are in `work/fwd_<k>/rows.jsonl`. Base generations
     are cached.
   - `read`: teacher-forced, up to {READ_TEXTS} texts (a recorded call plus a continuation: the recorded argument
     or any text you give), returning per-token log-probs and top-5 alternatives, and for up to {READ_LAYERS} layers
     the residual stream and MLP activations, and for up to {ATTN_LAYERS} layers attention probabilities, at the
     continuation positions; arrays in `work/fwd_<k>/text_<i>.npz`.
   - Add `"expect_edit": {file name: sha256}` to a request to make sure it runs with exactly the edit you mean;
     every `.done.json` records the edit snapshot it used.
3. **Data**: {DATA}

## Rules

- **No training of any kind:** no gradients, autograd, backward passes, optimisers or iterative fitting of weights
  to a loss. Closed-form computations from weights and activations (means, projections, SVD, least squares in
  closed form, thresholds you choose by reasoning) are allowed.
- **Run the model only through the forward-pass server.** Do not load the model for inference anywhere else.
- **Read only** the packet, your run folder, the model's config and tokenizer, and the model code in the Python
  environment. **Do not use the internet.** Do not try to reach other folders or processes. Every action you take
  is logged and audited; a breach invalidates the run.
- Before each code run or request, write one or two plain sentences in `{RUN_DIR}/notebook.md`: what you do and
  why. Keep the notebook as your working memory: hypotheses, evidence, decisions, current edit, open questions.

## How the run ends

Continue until all {BUDGET} passes are used, then make your final choice: the tensors saved in `{RUN_DIR}/edit/`
at that moment are your submission (an empty `edit/` is a valid choice if you believe no edit helps). Write
`{RUN_DIR}/final_answer.md`: the edit (modules, rank, formula), the mechanism you believe it addresses, the evidence
for and against, what you would test next, and the per-pass record. If your conversation must end before the budget
is used, first bring the notebook up to date (state, current edit, next step), so a fresh conversation can continue
from it.
