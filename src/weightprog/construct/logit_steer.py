"""Closed-form "gate + direction" edits on an MLP down_proj, and the predicted logit effect of any down_proj edit.

The construction (generalised from the study's best round-2 proposal, a rank-2 edit on layer 41's down_proj):

    W' = W + sum_r  alpha_r * d_r a_r^T          (B = [alpha_r d_r] [H, r], A = [a_r] [r, d_ff])

- a_r (a row of A) is a GATE over the layer's GeGLU activations y (the input of down_proj): a weighted ridge
  regression, fitted in closed form on captured activations, that is ~1 where the behaviour to change is about to
  happen and ~0 elsewhere (including at the hard negatives you care about).
- d_r (a column of B) is a DIRECTION in down_proj-output space, solved by least squares through the path to the
  logits so that one unit lowers some tokens' logits and raises others' by chosen amounts while holding a set of
  frequent tokens fixed. For the last layer the path is: post-FFN RMSNorm weight -> residual -> per-layer scalar ->
  final RMSNorm weight -> tied unembedding -> softcap. To first order

      dlogit_t ~= softcap'(l_t) * s / (rms(m) * rms(h)) * E_t . (w_final * w_post * delta)

  where m is down_proj's output at that position, h the captured (post-scalar) residual after layer L and s the
  layer scalar. `logit_direction` solves E_t . (w_final * w_post * d) = target_t (minimum norm); the bracketed factor
  is the per-position gain (`unit_gain`). The layer-scalar pitfall from the study: residual captures are
  post-scalar while the MLP output is added pre-scalar, so a simulation that adds the normed MLP change to a captured
  residual must multiply that change by s. Fable's first simulation did not and overstated the effect by 1/s
  (s = 0.48 on E4B layer 41: the observed change of the final residual was 0.48 x the predicted one, cosine 1.00).
  For the last layer s then cancels against the final norm's rms, so the logit effect itself does not depend on s.

`TailState` holds what is needed to recompute a layer's tail exactly from captured activations (resid_mid, mlp,
pli) and its weights. For the LAST layer, `predict_logit_shift(..., mode="exact")` is the true effect of the edit at
those positions (up to dtype); for earlier layers it is the direct-path (logit-lens) effect, which ignores every
later layer. mode="linear" is the first-order formula above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..edits import ModuleEdit
from ..names import module_name
from .linear import gelu_tanh, min_norm, ridge, rms_norm

PRE = "model.language_model."


@dataclass
class LogitPath:
    """Weights on the path from layer L's down_proj output to the logits."""

    layer: int
    w_post: np.ndarray            # post_feedforward_layernorm weight of layer L [H]
    w_final: np.ndarray           # final norm weight [H]
    scalar: float                 # layer_scalar of layer L
    E_rows: Callable[[Sequence[int]], np.ndarray]  # token ids -> unembedding rows [n, H]
    softcap: float | None = None
    eps: float = 1e-6

    @classmethod
    def from_weights(cls, get: Callable[[str], np.ndarray], layer: int, softcap: float | None, eps: float = 1e-6,
                     rows: Callable[[Sequence[int]], np.ndarray] | None = None) -> "LogitPath":
        E = None
        if rows is None:
            E = np.asarray(get(PRE + "embed_tokens.weight"), dtype=np.float64)
            rows = lambda ids: E[np.asarray(ids)]  # noqa: E731
        return cls(layer, np.asarray(get(f"{PRE}layers.{layer}.post_feedforward_layernorm.weight"), dtype=np.float64),
                   np.asarray(get(PRE + "norm.weight"), dtype=np.float64),
                   float(np.asarray(get(f"{PRE}layers.{layer}.layer_scalar")).reshape(-1)[0]), rows, softcap, eps)

    @classmethod
    def from_backend(cls, be, layer: int) -> "LogitPath":
        """Unembedding rows are read on demand (`be.read_rows`), never the whole [vocab, hidden] table."""
        name = PRE + "embed_tokens.weight"
        rows = (lambda ids: be.read_rows(name, list(ids)).astype(np.float64)) if hasattr(be, "read_rows") else None
        return cls.from_weights(be.read_weight, layer, be.spec.softcap, be.spec.rms_eps, rows=rows)

    @classmethod
    def from_reader(cls, reader, spec, layer: int) -> "LogitPath":
        """From a SafetensorsReader (no model load; unembedding rows read on demand)."""
        return cls.from_weights(reader.get, layer, spec.softcap, spec.rms_eps,
                                rows=lambda ids: reader.rows(PRE + "embed_tokens.weight", ids).astype(np.float64))


def logit_direction(path: LogitPath, targets: dict[int, float], ridge_eps: float = 1e-6) -> np.ndarray:
    """Minimum-norm d [H] with E_t . (w_final * w_post * d) = targets[t] for every listed token (unit gain)."""
    ids = list(targets)
    G = path.E_rows(ids).astype(np.float64) * (path.w_final * path.w_post)[None, :]
    return min_norm(G, np.array([targets[t] for t in ids]), ridge_eps)


def fit_gate(acts: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None, lam: float = 1.0,
             support: Sequence[int] | None = None, d_ff: int | None = None) -> np.ndarray:
    """Ridge gate over MLP activations: a [d_ff] with acts @ a ~= target. `support` restricts it to some units
    (the rest stay 0), e.g. the units that separate positive from negative rows best."""
    acts = np.asarray(acts, dtype=np.float64)
    d_ff = d_ff or acts.shape[1]
    a = np.zeros(d_ff)
    cols = np.arange(d_ff) if support is None else np.asarray(support)
    a[cols] = ridge(acts[:, cols], target, lam, weights)
    return a


def gate_direction_edit(layer: int, gates: np.ndarray, directions: np.ndarray, alpha=1.0, note: str = "",
                        proj: str = "down_proj") -> ModuleEdit:
    """Edit W' = W + sum_r alpha_r d_r a_r^T of `layers.{layer}.mlp.down_proj` (default) or another projection
    (`proj="o_proj"`: gates over the attention output, directions in the hidden space). gates [r, in],
    directions [r, out], alpha scalar or [r]."""
    A = np.atleast_2d(np.asarray(gates, dtype=np.float64))
    D = np.atleast_2d(np.asarray(directions, dtype=np.float64))
    al = np.broadcast_to(np.asarray(alpha, dtype=np.float64), (D.shape[0],))
    return ModuleEdit(module_name(layer, proj), (D * al[:, None]).T, A,
                      {"construct": "gate_direction", "rank": int(A.shape[0]), "alpha": al.tolist(), "note": note})


def residual_write_direction(be, layer: int, r: np.ndarray, block: str = "mlp") -> np.ndarray:
    """Module-output direction u [H] such that the block's contribution to the residual moves along the residual
    direction r [H] (Gemma 4 post-norms: residual += rms_norm(out) * w_post, so u = r / w_post, normalised).
    block="mlp" for down_proj (post_feedforward_layernorm), "attn" for o_proj (post_attention_layernorm). Use it to
    write a steering direction (e.g. a difference of means of `resid` captures) into any layer.
    Caution: dividing by w_post inflates the dimensions where it is tiny (down to about 0.001 on E4B), and the norm then
    rescales the whole output, so a large write lands mostly elsewhere. For kv_update values that must move the
    residual by a given amount, use `residual_write_value`, which solves under the norm constraint."""
    norm = "post_feedforward_layernorm" if block == "mlp" else "post_attention_layernorm"
    w = be.read_weight(f"{PRE}layers.{layer}.{norm}.weight").astype(np.float64)
    u = np.asarray(r, dtype=np.float64) / np.where(np.abs(w) < 1e-6, 1e-6, w)
    return u / np.linalg.norm(u)


def post_norm_reach(n: np.ndarray, scale: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """The normalised block output u (rms 1) whose residual contribution scale * u comes closest to
    scale * n + delta: min ||scale (u - n) - delta||^2 subject to ||u||^2 = len(u). n is the current normalised output
    (rms 1), scale the per-dimension factor the norm output is multiplied by (w_post, times layer_scalar if the target
    residual is post-scalar), delta the wanted residual change [H].

    Dividing delta by scale (as residual_write_direction does) blows up where scale is tiny (E4B w_post goes down to
    about 0.001), and the RMS norm then shrinks the whole output, so almost none of delta is written. Here the
    constraint is kept: Lagrange gives u = scale * b / (scale^2 + mu), b = delta + scale * n, with the scalar mu found by
    bisection on the norm constraint (monotone on mu > -min scale^2). Closed form plus a scalar search, no gradients."""
    n = np.asarray(n, dtype=np.float64)
    g = np.asarray(scale, dtype=np.float64)
    b = np.asarray(delta, dtype=np.float64) + g * n
    H = len(n)
    u_of = lambda mu: g * b / (g ** 2 + mu)  # noqa: E731
    f = lambda mu: float(np.sum(u_of(mu) ** 2)) - H  # noqa: E731
    lo = -float(np.min(g ** 2))
    hi = 1.0
    while f(hi) > 0:
        hi *= 2.0
    near = lo + 1e-9 * max(1.0, abs(lo))
    if f(near) <= 0:  # the constraint is met even next to the pole: take that point and rescale onto the sphere
        u = u_of(near)
        return u * np.sqrt(H / np.sum(u ** 2))
    lo = near
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return u_of(hi)


def residual_write_value(be, layer: int, mlp_out: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Values R [P, H] for `kv_update`: what layer `layer`'s down_proj must add at P rows so that the residual after
    the layer (the post-scalar "resid" capture) moves by `delta` ([H], or [P, H] per row), through Gemma 4's
    post-feedforward norm and layer scalar.

    mlp_out [P, H] is the captured raw down_proj output at those rows (`capture(..., what={"mlp_out"})`). Each row's
    new output keeps its RMS, so the post-norm sees exactly the solved direction (`post_norm_reach`). Rows that must
    stay unchanged get R = 0 (give them as keys with zero values). Ignored: the per-layer-input branch, which reads
    the layer's residual and so responds slightly; after building, measure the realised change with
    `backend.edits(...)` and `capture(..., what={"resid"})` against delta. On E4B layers 30-38, with delta the size of
    the version difference of means, this wrote the intended direction with cosine 0.96-0.99 (0.90-0.96 at twice that
    size); dividing by w_post had given 0.11 at layer 35."""
    mo = np.atleast_2d(np.asarray(mlp_out, dtype=np.float64))
    dl = np.broadcast_to(np.asarray(delta, dtype=np.float64), mo.shape)
    w = be.read_weight(f"{PRE}layers.{layer}.post_feedforward_layernorm.weight").astype(np.float64)
    names = set(be.weight_names())
    sk = f"{PRE}layers.{layer}.layer_scalar"
    s = float(np.asarray(be.read_weight(sk)).reshape(-1)[0]) if sk in names else 1.0
    eps = be.spec.rms_eps
    R = np.zeros_like(mo)
    for i, (m, d) in enumerate(zip(mo, dl)):
        rms = np.sqrt(np.mean(m ** 2) + eps)
        R[i] = rms * post_norm_reach(m / rms, s * w, d) - m
    return R


def cv_gate(acts: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None,
            lams: Sequence[float] = (0.1, 1.0, 10.0, 100.0), folds: int = 5, seed: int = 0, *,
            groups: Sequence[str] | None = None) -> dict:
    """Choose fit_gate's ridge `lam` by cross-validation: for each lam, out-of-fold gate values on every row, their
    AUC (target > 0.5 vs not) and the median out-of-fold gate on positives and |gate| on negatives. In-sample gate
    values say little: with tens of rows and thousands of units a ridge fits its own rows almost by construction.
    With `groups`, hold out complete groups to keep related versions of the same prompt in a single fold."""
    from ..probe import auc  # local import: probe imports construct
    X, y = np.asarray(acts, dtype=np.float64), np.asarray(target, dtype=np.float64)
    w = None if weights is None else np.asarray(weights, dtype=np.float64)
    if groups is None:
        ids = np.arange(len(y))
    else:
        if len(groups) != len(y):
            raise ValueError("groups must contain one label per row")
        _, ids = np.unique(groups, return_inverse=True)
    order = np.random.default_rng(seed).permutation(np.unique(ids))
    if not 2 <= folds <= len(order):
        raise ValueError("folds must be between two and the number of independent rows or groups")
    partitions = [np.flatnonzero(np.isin(ids, fold)) for fold in (order[k::folds] for k in range(folds))]
    out = {}
    for lam in lams:
        g = np.zeros(len(y))
        for te in partitions:
            tr = np.setdiff1d(np.arange(len(y)), te)
            a = ridge(X[tr], y[tr], lam, None if w is None else w[tr])
            g[te] = X[te] @ a
        pos = y > 0.5
        out[float(lam)] = {"auc": auc(g, pos) if pos.any() and (~pos).any() else None,
                           "median_pos": float(np.median(g[pos])) if pos.any() else None,
                           "median_abs_neg": float(np.median(np.abs(g[~pos]))) if (~pos).any() else None,
                           "p95_abs_neg": float(np.percentile(np.abs(g[~pos]), 95)) if (~pos).any() else None,
                           "oof_gate": g}
    return out


@dataclass
class TailState:
    """Captured inputs of layer L's MLP tail at P positions, plus the weights to recompute the rest exactly."""

    layer: int
    is_last: bool
    resid_mid: np.ndarray            # [P, H] residual the MLP output is added to
    act: np.ndarray                  # [P, d_ff] GeGLU activations entering down_proj
    pli: np.ndarray | None           # [P, Pd] per-layer input of layer L (None if the model has none)
    W_down: np.ndarray               # [H, d_ff]
    path: LogitPath
    pl_gate: np.ndarray | None = None    # per_layer_input_gate [Pd, H]
    pl_proj: np.ndarray | None = None    # per_layer_projection [H, Pd]
    pl_norm: np.ndarray | None = None    # post_per_layer_input_norm [H]
    extra: dict = field(default_factory=dict)
    captured_mlp_out: np.ndarray | None = None  # [P, H], for repeated predictions; stale if W_down changes after capture

    @classmethod
    def weights_from_backend(cls, be, layer: int) -> dict:
        """The layer's tail weights (read once; share them across many TailStates of the same layer)."""
        rw = be.read_weight
        L = f"{PRE}layers.{layer}."
        has_pl = any(n == L + "per_layer_input_gate.weight" for n in be.weight_names())
        return {"layer": layer, "is_last": layer == be.spec.n_layers - 1,
                "W_down": rw(L + "mlp.down_proj.weight").astype(np.float64), "path": LogitPath.from_backend(be, layer),
                "pl_gate": rw(L + "per_layer_input_gate.weight").astype(np.float64) if has_pl else None,
                "pl_proj": rw(L + "per_layer_projection.weight").astype(np.float64) if has_pl else None,
                "pl_norm": rw(L + "post_per_layer_input_norm.weight").astype(np.float64) if has_pl else None}

    @classmethod
    def from_backend(cls, be, ids, positions, layer: int, weights: dict | None = None) -> "TailState":
        w = weights or cls.weights_from_backend(be, layer)
        cap = be.capture(ids, positions, layers=[layer], what={"resid_mid", "mlp", "pli"})
        has_pl = "pli" in cap and w["pl_gate"] is not None
        return cls(layer, w["is_last"], cap["resid_mid"][0].astype(np.float64), cap["mlp"][0].astype(np.float64),
                   cap["pli"][0].astype(np.float64) if has_pl else None, w["W_down"], w["path"],
                   w["pl_gate"], w["pl_proj"], w["pl_norm"])

    def mlp_out(self) -> np.ndarray:
        """Return the captured raw MLP output when available, or recompute it from weights."""
        if self.captured_mlp_out is not None:
            return self.captured_mlp_out
        return self.act @ self.W_down.T

    def final_hidden(self, delta: np.ndarray | None = None) -> np.ndarray:
        """Residual after layer L (post scalar) with `delta` [P, H] added to down_proj's output."""
        eps = self.path.eps
        m = self.mlp_out() if delta is None else self.mlp_out() + delta
        h = self.resid_mid + rms_norm(m, self.path.w_post, eps)
        if self.pli is not None:
            g = gelu_tanh(h @ self.pl_gate.T) * self.pli
            h = h + rms_norm(g @ self.pl_proj.T, self.pl_norm, eps)
        return h * self.path.scalar

    def logits(self, token_ids: Sequence[int], delta: np.ndarray | None = None) -> np.ndarray:
        """Softcapped logits of token_ids [P, n] through the final norm (direct path for a non-last layer)."""
        hn = rms_norm(self.final_hidden(delta), self.path.w_final, self.path.eps)
        lg = hn @ self.path.E_rows(list(token_ids)).astype(np.float64).T
        c = self.path.softcap
        return np.tanh(lg / c) * c if c else lg


def edit_delta(state: TailState, edit: ModuleEdit) -> np.ndarray:
    """Change of down_proj's output at the captured positions: (act @ A^T) @ B^T, [P, H]."""
    return (state.act @ edit.A.astype(np.float64).T) @ edit.B.astype(np.float64).T


def gate_values(state: TailState, edit: ModuleEdit) -> np.ndarray:
    """a_r . y at each position, [P, r]."""
    return state.act @ edit.A.astype(np.float64).T


def predict_logit_shift(state: TailState, edit: ModuleEdit, token_ids: Sequence[int], mode: str = "exact") -> np.ndarray:
    """Predicted change of the softcapped logits of token_ids at the captured positions, [P, n].

    exact : recompute the tail of layer L (post-FFN norm, per-layer-input branch, scalar, final norm, unembedding,
            softcap) with and without the edit; the true effect for the last layer.
    linear: softcap'(l) * s / (rms(m) rms(h)) * E_t . (w_final * w_post * delta) (first order, no norm Jacobian terms
            and no per-layer-input feedback)."""
    if not edit.module.endswith(f"layers.{state.layer}.mlp.down_proj.weight"):
        raise ValueError(f"{edit.module} is not layer {state.layer}'s down_proj")
    d = edit_delta(state, edit)
    if mode == "exact":
        return state.logits(token_ids, d) - state.logits(token_ids)
    if mode != "linear":
        raise ValueError("mode is exact or linear")
    p, eps = state.path, state.path.eps
    m = state.mlp_out()
    h = state.final_hidden()
    rms_m = np.sqrt((m * m).mean(-1, keepdims=True) + eps)
    rms_h = np.sqrt((h * h).mean(-1, keepdims=True) + eps)
    Er = p.E_rows(list(token_ids)).astype(np.float64)
    raw = (d * (p.w_final * p.w_post)[None, :]) @ Er.T * (p.scalar / (rms_m * rms_h))
    if p.softcap:
        lg_raw = rms_norm(h, p.w_final, eps) @ Er.T
        raw = raw * (1.0 - np.tanh(lg_raw / p.softcap) ** 2)
    return raw


def unit_gain(state: TailState) -> np.ndarray:
    """Per-position gain s / (rms(m) rms(h_post)) [P] (= 1 / (rms(m) rms(h_pre))): a direction from logit_direction
    scaled by alpha moves the logits by about alpha * gain * gate (before the softcap's slope)."""
    eps = state.path.eps
    m, h = state.mlp_out(), state.final_hidden()
    return state.path.scalar / (np.sqrt((m * m).mean(-1) + eps) * np.sqrt((h * h).mean(-1) + eps))
