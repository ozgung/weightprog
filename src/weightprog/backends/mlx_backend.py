"""MLX backend (Apple silicon), ported from the study's forward-pass server and its MLX LoRA helpers.

Native generation and log-probs use mlx_lm's own Gemma 4 model with its KV caches. `capture` runs a manual forward
over the same modules (embedding scale, per-layer inputs, KV sharing, layer scalar, softcap), with blocked
sliding-window attention so a 15k-token prompt fits; it agrees with the native forward to float32 rounding on the
tiny test model (bf16 drifts more on long prompts: about 0.5 nats was seen on 14k-token prompts in the study).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..names import ModelSpec, canonical
from .base import SERVED, BackendBase, Force, Generate, SegmentResult, Tokenizer, check_capture_request, plan_segments

_DTYPES = {"bfloat16": "bfloat16", "bf16": "bfloat16", "float32": "float32", "fp32": "float32", "float16": "float16",
           "fp16": "float16"}


def _sliding_attention(mx, q, k, v, window: int, scale: float = 1.0):
    """Causal attention restricted to keys j with p - window < j <= p, in query blocks of `window` (memory O(L * 2W))."""
    B, H, L, D = q.shape
    KVH = k.shape[1]
    W = window
    sdpa = mx.fast.scaled_dot_product_attention
    if L <= W:
        return sdpa(q, k, v, scale=scale, mask="causal")
    nb = -(-L // W)
    pad = nb * W - L
    if pad:
        pw = [(0, 0), (0, 0), (0, pad), (0, 0)]
        q, k, v = mx.pad(q, pw), mx.pad(k, pw), mx.pad(v, pw)
    qb = q.reshape(B, H, nb, W, D).transpose(0, 2, 1, 3, 4).reshape(B * nb, H, W, D)

    def two_blocks(x):
        xb = x.reshape(B, KVH, nb, W, D)
        prev = mx.concatenate([mx.zeros_like(xb[:, :, :1]), xb[:, :, :-1]], axis=2)
        return mx.concatenate([prev, xb], axis=3).transpose(0, 2, 1, 3, 4).reshape(B * nb, KVH, 2 * W, D)

    kk, vv = two_blocks(k), two_blocks(v)
    a = mx.arange(W)[:, None]
    c = mx.arange(2 * W)[None]
    band = (c > a) & (c <= a + W)
    first = band & (c >= W)
    mask = mx.concatenate([first[None], mx.broadcast_to(band[None], (nb - 1, W, 2 * W))], axis=0)
    mask = mx.tile(mask[:, None], (B, 1, 1, 1)) if B > 1 else mask[:, None]
    o = sdpa(qb, kk, vv, scale=scale, mask=mask)
    o = o.reshape(B, nb, H, W, D).transpose(0, 2, 1, 3, 4).reshape(B, H, nb * W, D)
    return o[:, :, :L] if pad else o


class MLXBackend(BackendBase):
    name = "mlx"

    def __init__(self, model_dir: str | Path | None = None, dtype: str | None = "bfloat16", device: str | None = None):
        self.model = None
        if model_dir is not None:
            self.load(model_dir, dtype, device)

    # ---- loading -----------------------------------------------------------------------------------------
    def load(self, model_dir, dtype: str | None = "bfloat16", device: str | None = None):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm.utils import load_model

        self.mx = mx
        if device is not None:
            mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
        self.model_dir = Path(model_dir)
        self.spec = ModelSpec.from_config(self.model_dir)
        self.tokenizer = Tokenizer(self.model_dir)
        self.model, self.config = load_model(self.model_dir, lazy=False, strict=True)
        if dtype is not None:
            want = getattr(mx, _DTYPES[dtype])
            self.model.set_dtype(want)
        self.lm = self.model.language_model if hasattr(self.model, "language_model") else self.model
        self.tm = self.lm.model
        mx.eval(self.model.parameters())
        self.params = dict(tree_flatten(self.model.parameters()))
        self._pre = "language_model.model." if any(k.startswith("language_model.model.") for k in self.params) else "model."
        self.dtype = str(next(iter(self.params.values())).dtype)
        return self

    def _key(self, name: str) -> str:
        return self._pre + canonical(name).removeprefix("model.language_model.")

    def read_weight(self, name: str) -> np.ndarray:
        mx = self.mx
        k = self._key(name)
        if k not in self.params:
            raise KeyError(f"{name} (mlx key {k}) not in the model")
        a = self.params[k].astype(mx.float32)
        mx.eval(a)
        return np.array(a)

    def weight_names(self) -> list[str]:
        return sorted("model.language_model." + k.removeprefix(self._pre) for k in self.params)

    # ---- edits -------------------------------------------------------------------------------------------
    def _apply(self, edit_set) -> dict:
        mx = self.mx
        from mlx.utils import tree_unflatten

        originals = {}
        try:
            for e in edit_set:
                k = self._key(e.module)
                if k not in self.params:
                    raise KeyError(f"edit module {e.module} not in the model")
                W = self.params[k]
                D = mx.array(e.B.astype(np.float32)) @ mx.array(e.A.astype(np.float32))
                if tuple(D.shape) != tuple(W.shape):
                    raise ValueError(f"edit shape {tuple(D.shape)} != {tuple(W.shape)} for {e.module}")
                originals.setdefault(k, W)
                new = (W.astype(mx.float32) + D).astype(W.dtype)
                self.model.update(tree_unflatten([(k, new)]))
                self.params[k] = new
            mx.eval(self.model.parameters())
        except Exception:
            self._restore(originals)
            raise
        return originals

    def _restore(self, originals: dict) -> None:
        from mlx.utils import tree_unflatten

        if originals:
            self.model.update(tree_unflatten(list(originals.items())))
            self.params.update(originals)
            self.mx.eval(self.model.parameters())

    # ---- native forward ------------------------------------------------------------------------------------
    def _prefill(self, ids: Sequence[int]):
        mx = self.mx
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(self.model)
        for s in range(0, len(ids), self.prefill_chunk):
            self.model(mx.array(list(ids[s:s + self.prefill_chunk]))[None], cache=cache)
            mx.eval([c.state for c in cache])
        return cache

    def _sampler(self, greedy: bool, sampling):
        from mlx_lm.sample_utils import make_sampler

        if greedy:
            return make_sampler(temp=0.0)
        return make_sampler(temp=sampling.temperature, top_p=sampling.top_p, top_k=sampling.top_k)

    def decode_segments(self, ids, segments, draws=1, seeds=None, sampling=SERVED, abort_on_unclosed=True):
        mx = self.mx
        plan_segments(segments)
        ids = [int(t) for t in ids]
        if not ids:
            raise ValueError("empty prompt")
        seeds = self._seeds(draws, seeds)
        cache = self._prefill(ids[:-1])
        outs = []
        for d in range(draws):
            sampler = self._sampler(d == 0, sampling)
            mx.random.seed(seeds[d])
            c = copy.deepcopy(cache)
            pending, last_stop, aborted, res = ids[-1:], None, False, []
            for seg in segments:
                if aborted:
                    res.append(SegmentResult(skipped=True))
                    continue
                if isinstance(seg, Force):
                    pending = pending + list(seg.ids)
                    res.append(SegmentResult(tokens=list(seg.ids), closed=1))
                    continue
                if not pending:
                    if last_stop is None:
                        raise ValueError("a Generate segment needs input: put a Force segment before it")
                    pending = [last_stop]
                logits = self.model(mx.array(pending)[None], cache=c)[:, -1, :]
                pending = []
                out, closed, stop = [], 0, None
                stops = set(seg.stop_ids)
                for _ in range(seg.max_new):
                    t = int(sampler(logits - mx.logsumexp(logits, axis=-1, keepdims=True)).item())
                    if t in stops:
                        closed, stop, last_stop = 1, t, t
                        break
                    out.append(t)
                    logits = self.model(mx.array([[t]]), cache=c)[:, -1, :]
                res.append(SegmentResult(tokens=out, closed=closed, stop_id=stop))
                if not closed:
                    last_stop = None
                    if abort_on_unclosed:
                        aborted = True
            outs.append(res)
        return outs

    def logprobs(self, ids, positions) -> np.ndarray:
        """Log-softmax at `positions` from mlx_lm's cached forward (the generation code path)."""
        mx = self.mx
        ids = [int(t) for t in ids]
        pos = self._positions(positions)
        p0, p1 = min(pos), max(pos)
        if p1 >= len(ids):
            raise ValueError("position beyond the sequence")
        cache = self._prefill(ids[:p0]) if p0 else None
        if cache is None:
            from mlx_lm.models.cache import make_prompt_cache
            cache = make_prompt_cache(self.model)
        lg = self.model(mx.array(ids[p0:p1 + 1])[None], cache=cache)[0].astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        sel = mx.array([p - p0 for p in pos])
        out = lp[sel]
        mx.eval(out)
        return np.array(out)

    # ---- manual forward with captures -----------------------------------------------------------------------
    def _attn(self, att, x, shared_kv, window):
        mx = self.mx
        B, L, _ = x.shape
        q = att.q_norm(att.q_proj(x).reshape(B, L, att.n_heads, att.head_dim))
        if shared_kv is not None:
            k, v = shared_kv
        else:
            k = att.k_proj(x).reshape(B, L, att.n_kv_heads, att.head_dim)
            v = k if att.use_k_eq_v else att.v_proj(x).reshape(B, L, att.n_kv_heads, att.head_dim)
            k = att.rope(att.k_norm(k).transpose(0, 2, 1, 3), offset=0)
            v = att.v_norm(v).transpose(0, 2, 1, 3)
        q = att.rope(q.transpose(0, 2, 1, 3), offset=0)
        if att.is_sliding:
            o = _sliding_attention(mx, q, k, v, window, att.scale)
        else:
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=att.scale, mask="causal")
        return att.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1)), (k, v), q

    def capture(self, ids, positions, layers=(), what=("resid",), attn_layers=()) -> dict:
        mx = self.mx
        import mlx.nn as nn
        from mlx_lm.models.gemma4_text import geglu

        what = check_capture_request(what)
        layers = sorted({int(i) for i in layers})
        attn_layers = sorted({int(i) for i in attn_layers}) if "attn" in what else []
        ids = [int(t) for t in ids]
        pos_l = self._positions(positions)
        if max(pos_l) >= len(ids):
            raise ValueError("position beyond the sequence")
        tm = self.tm
        n = len(tm.layers)
        for i in layers + attn_layers:
            if not 0 <= i < n:
                raise ValueError(f"layer {i} out of range")
        x_ids = mx.array(ids)[None]
        emb = tm.embed_tokens(x_ids)
        h = emb * tm.embed_scale
        pli_all = tm._project_per_layer_inputs(h, tm._get_per_layer_inputs(x_ids, emb)) if tm.hidden_size_per_layer_input else None
        prev = list(tm.previous_kvs)
        kvs = {}
        pos = mx.array(pos_l)
        L = len(ids)
        W = tm.window_size
        keep = {k: [] for k in ("resid", "resid_pre", "resid_mid", "mlp", "mlp_out", "pli", "attn")}
        f32 = lambda a: np.array(a.astype(mx.float32))  # noqa: E731
        for i, layer in enumerate(tm.layers):
            if getattr(layer, "enable_moe", False):
                raise NotImplementedError("MoE layers are not supported")
            want = i in layers
            pli = pli_all[:, :, i, :] if pli_all is not None else None
            if want and "resid_pre" in what:
                keep["resid_pre"].append(f32(h[0, pos]))
            shared = kvs.get(prev[i]) if prev[i] != i else None
            residual = h
            xa = layer.input_layernorm(h)
            a_out, kv, q_all = self._attn(layer.self_attn, xa, shared, W)
            if i in attn_layers:
                att = layer.self_attn
                q = q_all[:, :, pos, :].astype(mx.float32)
                k = kv[0]
                k = mx.repeat(k, att.n_heads // k.shape[1], axis=1).astype(mx.float32)
                s = (q @ k.transpose(0, 1, 3, 2)) * att.scale
                j = mx.arange(L)[None, :]
                p_ = pos[:, None]
                allowed = (j <= p_) & ((j > p_ - W) if att.is_sliding else (j >= 0))
                s = mx.where(allowed[None, None], s, -1e30)
                keep["attn"].append(np.array(mx.softmax(s, axis=-1)[0]))
            if prev[i] == i:
                kvs[i] = kv
            h = residual + layer.post_attention_layernorm(a_out)
            if want and "resid_mid" in what:
                keep["resid_mid"].append(f32(h[0, pos]))
            xm = layer.pre_feedforward_layernorm(h)
            act = geglu(layer.mlp.gate_proj(xm), layer.mlp.up_proj(xm))
            if want and "mlp" in what:
                keep["mlp"].append(f32(act[0, pos]))
            mo = layer.mlp.down_proj(act)
            if want and "mlp_out" in what:
                keep["mlp_out"].append(f32(mo[0, pos]))
            h = h + layer.post_feedforward_layernorm(mo)
            if layer.per_layer_input_gate is not None and pli is not None:
                if want and "pli" in what:
                    keep["pli"].append(f32(pli[0, pos]))
                g = nn.gelu_approx(layer.per_layer_input_gate(h)) * pli
                h = h + layer.post_per_layer_input_norm(layer.per_layer_projection(g))
            if layer.layer_scalar is not None:
                h = h * layer.layer_scalar
            if want and "resid" in what:
                keep["resid"].append(f32(h[0, pos]))
            mx.eval(h)
        out = {"positions": np.array(pos_l), "layers": np.array(layers, dtype=np.int64),
               "attn_layers": np.array(attn_layers, dtype=np.int64)}
        for k, v in keep.items():
            if v:
                out[k] = np.stack(v)
        if "logits" in what:
            hn = tm.norm(h[0, pos])
            lg = (tm.embed_tokens.as_linear(hn) if getattr(self.lm, "tie_word_embeddings", True) else self.lm.lm_head(hn)).astype(mx.float32)
            sc = self.lm.final_logit_softcapping
            if sc:
                lg = mx.tanh(lg / sc) * sc
            out["logits"] = np.array(lg)
        return out
