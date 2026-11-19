# Methods

weightprog packages methods known from the model-editing and interpretability literature for an agent to apply to its
own target, and adds the evaluation around them. "On real E4B" says where a method has run on the real model, not
only on the tiny random Gemma 4 models of the tests; the rest is tested offline only.

| Method | In weightprog | From | On real E4B |
|---|---|---|---|
| **Find the behaviour** | | | |
| Activation capture at decision tokens | `backend.capture`, `probe.divergence`, `probe.decision_from_text`, `probe.collect_tail` | TransformerLens, nnsight, pyvene | checked against native log-probs ([validation](real-model-validation.md), check 4) |
| Linear probes and read-outs | `probe.readout`, `probe.layer_sweep`, `construct.cv_gate` | Alain and Bengio, [2016](https://arxiv.org/abs/1610.01644) | used by agents in the usability runs |
| Difference of means | `probe.layer_sweep` (norm of the difference), captures | Zou et al., [2023](https://arxiv.org/abs/2310.01405) | tiny models only (the layer-30 edit computed it in its own script) |
| Logit lens, direct logit attribution | `construct.logit_direction`, `LogitPath` | nostalgebraist, [2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens); Elhage et al., [2021](https://transformer-circuits.pub/2021/framework/index.html) | used by agents in the usability runs |
| Exact prediction of an edit's effect | `construct.predict_logit_shift`, `TailState` (last layer, free); `probe.measure_logit_shift` (any layer, two passes) | this library | checked ([validation](real-model-validation.md), check 3) |
| Activation patching, causal tracing | `probe.patch_effect`, `probe.patch_sweep` | Vig et al., [2020](https://arxiv.org/abs/2004.12265); Meng et al., [2022](https://arxiv.org/abs/2202.05262) | used by one agent run (GPT-6.2 Sol) |
| Attention and head analysis | attention probabilities in `capture` (building blocks only) | Olsson et al., [2022](https://arxiv.org/abs/2209.11895) | tiny models only |
| Sparse autoencoders | not in the library | Bricken et al., [2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html) | |
| **Change it** | | | |
| Unit (neuron) patching: rewire low-use MLP units into detector-writers | `construct.quiet_units`, `mlp_input`, `unit_patch` | Dai et al., [2022](https://arxiv.org/abs/2104.08696); Huang et al., [2023](https://arxiv.org/abs/2301.09785) | the judged edits of the library runs ([usability tests](usability-tests.md)) |
| Gated logit-direction edit | `construct.fit_gate`, `cv_gate`, `gate_direction_edit` | cf. Transformer-Patcher (Huang et al., [2023](https://arxiv.org/abs/2301.09785)) | tried by agents in the usability runs; no judged edit |
| Key -> value update with covariance whitening | `construct.second_moment`, `kv_update`, `residual_write_value` | ROME (Meng et al., [2022](https://arxiv.org/abs/2202.05262)); MEMIT (Meng et al., [2023](https://arxiv.org/abs/2210.07229)) | one judged edit at layer 30 (Nov 2026): it moved the target on held-out cases and failed the keep check ([validation](real-model-validation.md)); also tried by an agent in the usability runs |
| Steering direction written into the weights | `construct.residual_write_direction` (direction), `residual_write_value` (values for a kv update) | Turner et al., [2023](https://arxiv.org/abs/2308.10248); Zou et al., [2023](https://arxiv.org/abs/2310.01405) | the `residual_write_value` solve (then in the experiment's own script, same values) built that layer-30 edit; measured write cosine 0.96-0.99 at layers 30-38 for writes of the natural size, 0.90-0.96 at twice that |
| Null-space restriction | `construct.null_space_basis`, `project_edit`, `kv_update(..., null_basis=U)` | AlphaEdit (Fang et al., [2025](https://arxiv.org/abs/2410.02355)) | tiny models only |
| Directional ablation | `construct.ablate_output`, `ablate_input` | Arditi et al., [2024](https://arxiv.org/abs/2406.11717) | tiny models only |
| Attention-side edits | `gate_direction_edit(..., proj="o_proj")`, ablation of `o_proj` (no query, key or value edits) | | tiny models only |
| Scaling and merging edits | `ModuleEdit.scaled`, `EditSet.scaled`, `EditSet.merge` | task arithmetic (Ilharco et al., [2023](https://arxiv.org/abs/2212.04089)) | tiny models only |
| **Check and ship it** | | | |
| Matched random-edit controls, keep cases, KL and NLL side effects | `controls.matched_control`, `evaluate.run`, `side_effects`, `compare` | this library | every judged edit above |
| Export as a LoRA adapter | `edits.export_peft` | LoRA (Hu et al., [2022](https://arxiv.org/abs/2106.09685)) | adapters served by vLLM in the study |

Weight-space programming by language-model proposers, the idea of having a language model write weights directly, is
prior work (Takehi et al., see the README's Credit section).
