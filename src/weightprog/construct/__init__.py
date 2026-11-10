"""Closed-form edit constructors (no gradients): gate + direction edits, a covariance-whitened key -> value update
(the ROME/MEMIT closed form), direction ablation, residual-direction writes for any layer, gate cross-validation, and
unit patching (rewire low-use MLP units), null-space projection (AlphaEdit), and the exact predicted logit effect of a
last-layer down_proj edit."""

from .ablation import ablate_input, ablate_output, quiet_units
from .linear import gelu_tanh, min_norm, orthonormal, ridge, rms_norm
from .covariance import kv_update, second_moment
from .units import mlp_input, null_space_basis, project_edit, unit_patch
from .logit_steer import (LogitPath, TailState, cv_gate, edit_delta, fit_gate, gate_direction_edit, gate_values,
                          logit_direction, predict_logit_shift, residual_write_direction, unit_gain)

__all__ = ["LogitPath", "TailState", "ablate_input", "ablate_output", "cv_gate", "edit_delta", "fit_gate",
           "gate_direction_edit", "gate_values", "gelu_tanh", "kv_update", "logit_direction", "min_norm", "orthonormal",
           "mlp_input", "null_space_basis", "predict_logit_shift", "project_edit", "quiet_units",
           "residual_write_direction", "ridge", "rms_norm", "second_moment", "unit_gain", "unit_patch"]
