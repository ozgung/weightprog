# Other models

weightprog is written for Gemma 4. Some parts depend on the model and some do not:

- **Model-independent** (NumPy on arrays): the edit format (`EditSet`, `W + B A` in `.npz` files), matched random
  controls, cases, `evaluate.compare` and `Criteria`, and the linear-algebra constructors (`ridge`, `second_moment`,
  `kv_update`, `null_space_basis`, `project_edit`, ablation).
- **Gemma 4-specific:**
  - The backends load Gemma 4 classes, and `capture` recomputes Gemma 4's layer: post-norms, per-layer inputs,
    `layer_scalar`, KV sharing and the softcap.
  - `names.py` hard-codes the tensor names (`model.language_model.layers...`) and reads `ModelSpec` from a Gemma 4
    config.
  - These constructors encode the architecture: `TailState` / `predict_logit_shift`, `logit_direction` (tied
    embeddings, final norm), `unit_patch` (GeGLU), and `residual_write_value` / `residual_write_direction`
    (post-norm).

What to do:

- **Another dense Gemma 4 model (E2B, 31B):** it will probably load and run as is. Only E4B has been checked on the
  real model, so first run the checks in [real-model-validation.md](real-model-validation.md) on yours,
  above all that `capture` matches the native log-probabilities. Mixture-of-experts models (Gemma 4 26B A4B) are not
  supported.
- **Another architecture (Llama, Qwen, Mistral, ...):**
  - Write a backend that implements `backends.base.Backend` (the existing ones are about 340 lines each).
  - Map its tensor names. `names.py` is not yet parameterised by backend, so this also needs a change there.
  - Evaluation, controls, `kv_update` and the null-space tools then work unchanged.
  - Adapt the Gemma-specific constructors. Most models make them simpler: no post-norm, so a residual write is the
    value itself; no softcap; a separate `lm_head`.
  - Before trusting any measurement, check that `capture` matches the native forward pass.
