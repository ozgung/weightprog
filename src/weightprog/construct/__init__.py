"""Closed-form edit constructors (no gradients): direction ablation, gate + direction down_proj edits, and the
predicted logit effect of a down_proj edit."""

from .ablation import ablate_input, ablate_output, quiet_units
from .linear import gelu_tanh, min_norm, orthonormal, ridge, rms_norm
from .logit_steer import (LogitPath, TailState, edit_delta, fit_gate, gate_direction_edit, gate_values, logit_direction,
                          predict_logit_shift, unit_gain)

__all__ = ["LogitPath", "TailState", "ablate_input", "ablate_output", "edit_delta", "fit_gate", "gate_direction_edit",
           "gate_values", "gelu_tanh", "logit_direction", "min_norm", "orthonormal", "predict_logit_shift", "quiet_units",
           "ridge", "rms_norm", "unit_gain"]
