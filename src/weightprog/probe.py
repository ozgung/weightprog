"""Find where a behaviour is decided and read the model there, across many examples.

    from weightprog.probe import show_tokens, divergence, margins, collect_tail, measure_logit_shift, readout

    show_tokens(tok, "user.model_dump()")                 # [(id, piece), ...]: see how text is split
    d = divergence(tok, prompt_text, "model_dump()", "dict()")
    # d.prefix_ids: prompt + the tokens both continuations share; d.good_id / d.bad_id: the first tokens that differ
    m = margins(be, [(d.prefix_ids, d.good_id, d.bad_id), ...])    # log P(good) - log P(bad) at the decision, [N]
    st = collect_tail(be, [(ids, [pos]), ...], layer=41)           # one TailState over many sequences
    shift = measure_logit_shift(be, edit, ids, positions, [good_id, bad_id])   # native, any layer, [P, n]
    auc = readout(acts, labels)                                    # held-out AUC of a ridge read-out

A "decision position" is the position whose next-token distribution chooses between the behaviours: the last
token of `d.prefix_ids`. Capture there, fit gates there, and measure margins there before spending generations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .construct.linear import ridge
from .construct.logit_steer import TailState


def show_tokens(tokenizer, text: str) -> list[tuple[int, str]]:
    """Token ids and their decoded pieces."""
    return [(i, tokenizer.decode([i])) for i in tokenizer.encode(text)]


@dataclass
class Divergence:
    prefix_ids: list[int]      # prompt plus the tokens both continuations share
    good_id: int               # first token of the good continuation after the shared part
    bad_id: int                # first token of the bad continuation after the shared part
    good_rest: list[int]       # the good continuation's tokens from good_id on
    bad_rest: list[int]


def divergence(tokenizer, prompt: str, good: str, bad: str) -> Divergence:
    """Tokenise prompt+good and prompt+bad together (so merges across the boundary are real) and split at the first
    token where they differ. Raises if one continuation is a token prefix of the other."""
    a, b = tokenizer.encode(prompt + good), tokenizer.encode(prompt + bad)
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    if n == len(a) or n == len(b):
        raise ValueError("one continuation is a token prefix of the other")
    return Divergence(a[:n], a[n], b[n], a[n:], b[n:])


def decision_from_text(tokenizer, prompt: str, generated: str, pattern: str, group: int = 0) -> Divergence | None:
    """Locate a behaviour inside the model's own generation (e.g. a census row): the token where `pattern` (its
    `group`) starts in `generated`. Returns a Divergence whose prefix_ids end right before that token and whose
    bad_id is that token (good_id = -1, fill it yourself), or None if the pattern does not occur. Use it to harvest
    positives (bad generations of target cases) and hard negatives (good generations of keep cases) at the rows the
    model really decides on, instead of hand-written prefills only. The text is re-tokenised, so a merge across the
    boundary can shift the row by a token: check with show_tokens."""
    import re

    m = re.search(pattern, generated)
    if not m:
        return None
    start = len(prompt) + m.start(group)
    ids = tokenizer.encode(prompt + generated)
    pos = 0
    for k, t in enumerate(ids):
        piece = tokenizer.decode([t])
        if pos + len(piece) > start:
            return Divergence(ids[:k], -1, t, [], ids[k:])
        pos += len(piece)
    return None


def layer_sweep(backend, positives: Sequence[Sequence[int]], negatives: Sequence[Sequence[int]],
                layers: Sequence[int] | None = None, what: str = "resid", lam: float = 10.0) -> list[dict]:
    """Where is the choice readable? For each layer: cross-validated AUC of a ridge read-out separating the last
    position of `positives` from that of `negatives` (token id lists), and the norm of the difference of means
    (relative to the mean norm). Pick layers where the AUC is high for gates; the difference of means is a
    steering/ablation direction candidate (see construct.residual_write_direction)."""
    layers = list(layers) if layers is not None else list(range(backend.spec.n_layers))
    caps = [backend.capture(list(ids), [len(ids) - 1], layers=layers, what={what})[what][:, 0]
            for ids in list(positives) + list(negatives)]
    X = np.stack(caps)                                   # [N, n_layers, width]
    y = np.r_[np.ones(len(positives)), np.zeros(len(negatives))]
    out = []
    for j, L in enumerate(layers):
        A = X[:, j].astype(np.float64)
        dm = A[y == 1].mean(0) - A[y == 0].mean(0)
        out.append({"layer": L, "auc": readout(A, y, lam=lam, folds=min(5, int(min(y.sum(), (1 - y).sum())))),
                    "mean_diff_rel_norm": float(np.linalg.norm(dm) / np.mean(np.linalg.norm(A, axis=1))),
                    "mean_diff": dm})
    return out


def patch_effect(backend, clean_ids: Sequence[int], corrupt_ids: Sequence[int], layer: int,
                 positions: Sequence[int], good_id: int, bad_id: int) -> dict:
    """Activation patching (causal tracing): run the corrupted prompt with the residual stream after `layer` at
    `positions` replaced by the clean prompt's, and read log P(good) - log P(bad) at the last position.
    `restored` = (patched - corrupt) / (clean - corrupt): 1 means those activations carry the whole difference, 0 none.
    Positions index both sequences, so use prompts of equal length that differ only where you vary the cause (e.g.
    "US English" vs "British English" padded to the same token count)."""
    if len(clean_ids) != len(corrupt_ids):
        raise ValueError("clean and corrupt prompts must have the same token length")
    last = len(clean_ids) - 1
    pos = [int(p) for p in positions]
    cl = backend.capture(list(clean_ids), sorted(set(pos + [last])), layers=[layer], what={"resid", "logits"})
    clean_res = {p: cl["resid"][0][sorted(set(pos + [last])).index(p)] for p in pos}
    li = sorted(set(pos + [last])).index(last)
    m_clean = _margin_from_logits(cl["logits"][li], good_id, bad_id)
    co = backend.capture(list(corrupt_ids), [last], what={"logits"})
    m_corrupt = _margin_from_logits(co["logits"][0], good_id, bad_id)
    pa = backend.capture(list(corrupt_ids), [last], what={"logits"}, patch_resid={layer: clean_res})
    m_patched = _margin_from_logits(pa["logits"][0], good_id, bad_id)
    den = m_clean - m_corrupt
    return {"layer": layer, "positions": pos, "clean": m_clean, "corrupt": m_corrupt, "patched": m_patched,
            "restored": (m_patched - m_corrupt) / den if abs(den) > 1e-9 else float("nan")}


def patch_sweep(backend, clean_ids, corrupt_ids, good_id: int, bad_id: int, layers: Sequence[int] | None = None,
                positions: Sequence[int] | None = None) -> list[dict]:
    """patch_effect for every layer (default) at `positions` (default: every position where the two prompts differ,
    and the last position separately). Where `restored` jumps from ~0 to ~1 is where the information is written."""
    layers = list(layers) if layers is not None else list(range(backend.spec.n_layers))
    if positions is None:
        diff = [i for i, (a, b) in enumerate(zip(clean_ids, corrupt_ids)) if a != b]
        groups = [("differing tokens", diff), ("last position", [len(clean_ids) - 1])]
    else:
        groups = [("given positions", list(positions))]
    out = []
    for name, pos in groups:
        for L in layers:
            r = patch_effect(backend, clean_ids, corrupt_ids, L, pos, good_id, bad_id)
            r["where"] = name
            out.append(r)
    return out


def _margin_from_logits(logits_row: np.ndarray, good_id: int, bad_id: int) -> float:
    z = np.asarray(logits_row, dtype=np.float64)
    lse = np.log(np.exp(z - z.max()).sum()) + z.max()
    return float((z[good_id] - lse) - (z[bad_id] - lse))


def margins(backend, items: Sequence[tuple[Sequence[int], int, int]]) -> np.ndarray:
    """log P(good_id) - log P(bad_id) predicted from the last position of each ids (native cached forward)."""
    out = []
    for ids, g, b in items:
        lp = backend.logprobs(list(ids), [len(ids) - 1])[0]
        out.append(float(lp[g] - lp[b]))
    return np.array(out)


def collect_tail(backend, items: Sequence[tuple[Sequence[int], Sequence[int]]], layer: int) -> TailState:
    """Capture layer `layer`'s MLP tail at the given positions of several sequences and stack them into one
    TailState (rows in input order). Use it to fit gates and predict an edit's effect over many examples."""
    w = TailState.weights_from_backend(backend, layer)          # weights read once, shared by every row
    states = [TailState.from_backend(backend, list(ids), list(pos), layer, weights=w) for ids, pos in items]
    s0 = states[0]
    return TailState(s0.layer, s0.is_last, np.concatenate([s.resid_mid for s in states]),
                     np.concatenate([s.act for s in states]),
                     None if s0.pli is None else np.concatenate([s.pli for s in states]),
                     s0.W_down, s0.path, s0.pl_gate, s0.pl_proj, s0.pl_norm,
                     {"n_per_item": [len(s.act) for s in states]})


def measure_logit_shift(backend, edit_set, ids: Sequence[int], positions: Sequence[int],
                        token_ids: Sequence[int]) -> np.ndarray:
    """Measured change of log-probabilities of token_ids at positions under the edit, [P, n]. Works for an edit in
    any layer (it runs the model twice); `construct.predict_logit_shift` is the free, exact shortcut for the last
    layer's down_proj only."""
    t = list(token_ids)
    lp0 = backend.logprobs(list(ids), list(positions))[:, t]
    with backend.edits(edit_set):
        lp1 = backend.logprobs(list(ids), list(positions))[:, t]
    return lp1 - lp0


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve (probability a positive scores above a negative; ties count half)."""
    s, y = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        raise ValueError("need both classes")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def readout(acts: np.ndarray, labels: np.ndarray, lam: float = 1.0, folds: int = 5, seed: int = 0) -> float:
    """Cross-validated AUC of a ridge read-out of `labels` (0/1) from `acts` [N, d]: is the coming choice linearly
    readable at this layer and position? Near 0.5 means no; near 1 means a gate can find it."""
    X, y = np.asarray(acts, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    idx = np.random.default_rng(seed).permutation(len(y))
    scores = np.zeros(len(y))
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te)
        mu = X[tr].mean(0)
        w = ridge(X[tr] - mu, y[tr] - y[tr].mean(), lam)
        scores[te] = (X[te] - mu) @ w
    return auc(scores, y)
