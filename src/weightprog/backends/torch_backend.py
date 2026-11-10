"""PyTorch backend (CUDA, CPU; MPS untested), on Hugging Face transformers' Gemma 4 classes.

Ported from the study's `wsp_torch.py` and extended with attention-probability capture, the extra capture kinds and
the generic multi-segment decode. Native generation and log-probs use the HF model with its DynamicCache; `capture`
runs a manual forward over the same HF modules (query-chunked attention, so long prompts fit), which agrees with the
native forward to float32 rounding on the tiny test model.

The sampler reproduces mlx_lm's semantics (top-p on the untempered distribution, then top-k, then temperature) with
torch's RNG, so the sampled distribution is the same as the MLX backend's but sampled tokens are not. (The study's
wsp_torch.py applied temperature before top-k/top-p; this port does not.)
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from ..names import ModelSpec, canonical
from .base import SERVED, BackendBase, Force, SegmentResult, Tokenizer, check_capture_request, plan_segments

_DT = {"bfloat16": "bfloat16", "bf16": "bfloat16", "float32": "float32", "fp32": "float32", "float16": "float16",
       "fp16": "float16"}


class TorchBackend(BackendBase):
    name = "torch"

    def __init__(self, model_dir=None, dtype: str | None = "bfloat16", device: str | None = None, attn: str = "sdpa",
                 query_chunk: int = 2048):
        self.model = None
        self.attn_impl = attn
        self.query_chunk = query_chunk
        if model_dir is not None:
            self.load(model_dir, dtype, device)

    # ---- loading -----------------------------------------------------------------------------------------
    def load(self, model_dir, dtype: str | None = "bfloat16", device: str | None = None):
        import torch
        import transformers

        self.torch = torch
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.spec = ModelSpec.from_config(self.model_dir)
        self.tokenizer = Tokenizer(self.model_dir)
        cfg = json.loads((self.model_dir / "config.json").read_text())
        tdt = getattr(torch, _DT[dtype or cfg.get("dtype") or cfg.get("torch_dtype") or "bfloat16"])
        if cfg.get("model_type") == "gemma4":
            impl = {"text_config": self.attn_impl}
            for sub in ("vision_config", "audio_config"):
                if cfg.get(sub):
                    impl[sub] = "eager"  # the audio tower has no SDPA path; towers are dropped below anyway
            cls = transformers.Gemma4ForConditionalGeneration
        else:
            impl = self.attn_impl
            cls = transformers.Gemma4ForCausalLM
        self.model = cls.from_pretrained(self.model_dir, dtype=tdt, attn_implementation=impl).to(self.device).eval()
        inner = self.model.model
        for tower in ("audio_tower", "vision_tower", "embed_audio", "embed_vision"):
            if getattr(inner, tower, None) is not None:
                setattr(inner, tower, None)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        self.lm = inner.language_model if hasattr(inner, "language_model") else inner
        self.text_config = self.lm.config
        self.params = dict(self.model.named_parameters())
        self.params.update(dict(self.model.named_buffers()))
        self._pre = "model.language_model." if any(k.startswith("model.language_model.") for k in self.params) else "model."
        self.dtype = str(tdt).removeprefix("torch.")
        return self

    def _key(self, name: str) -> str:
        return self._pre + canonical(name).removeprefix("model.language_model.")

    def _param(self, name: str):
        k = self._key(name)
        if k not in self.params:
            raise KeyError(f"{name} (torch key {k}) not in the model")
        return self.params[k]

    def read_weight(self, name: str) -> np.ndarray:
        return self._param(name).detach().float().cpu().numpy().copy()  # never a view of the live parameter

    def read_rows(self, name: str, rows) -> np.ndarray:
        """float32 copy of some rows of a parameter (e.g. unembedding rows) without copying the whole tensor."""
        idx = self.torch.as_tensor(np.asarray(rows, dtype=np.int64), device=self._param(name).device)
        return self._param(name).detach()[idx].float().cpu().numpy().copy()

    def weight_names(self) -> list[str]:
        out = []
        for k in self.params:
            if k.startswith(self._pre) and not k.endswith(("_inv_freq", "embed_scale")) and ".rotary_emb." not in k:
                out.append("model.language_model." + k.removeprefix(self._pre))
        return sorted(set(out))

    # ---- edits -------------------------------------------------------------------------------------------
    def _apply(self, edit_set) -> dict:
        torch = self.torch
        originals = {}
        try:
            with torch.no_grad():
                for e in edit_set:
                    p = self._param(e.module)
                    D = torch.from_numpy(e.B.astype(np.float32)).to(self.device) @ torch.from_numpy(e.A.astype(np.float32)).to(self.device)
                    if tuple(D.shape) != tuple(p.shape):
                        raise ValueError(f"edit shape {tuple(D.shape)} != {tuple(p.shape)} for {e.module}")
                    originals.setdefault(e.module, p.data.clone())
                    p.data.copy_((p.data.float() + D).to(p.dtype))
        except Exception:
            self._restore(originals)
            raise
        return originals

    def _restore(self, originals: dict) -> None:
        with self.torch.no_grad():
            for name, w in originals.items():
                self._param(name).data.copy_(w)

    # ---- native forward ------------------------------------------------------------------------------------
    def _forward(self, ids, cache, start: int, logits_to_keep=0):
        torch = self.torch
        x = torch.tensor([list(ids)], device=self.device)
        out = self.model(input_ids=x, past_key_values=cache, use_cache=True,
                         cache_position=torch.arange(start, start + len(ids), device=self.device),
                         logits_to_keep=logits_to_keep)
        return out.logits, out.past_key_values

    def _prefill(self, ids):
        cache = None
        for s in range(0, len(ids), self.prefill_chunk):
            _, cache = self._forward(ids[s:s + self.prefill_chunk], cache, s, logits_to_keep=1)
        return cache

    def _sample(self, logits, gen, sampling):
        """mlx_lm order: top-p on the T=1 distribution, then top-k, then categorical at temperature T."""
        torch = self.torch
        lp = logits.float() - torch.logsumexp(logits.float(), -1, keepdim=True)
        top_p, top_k, T = sampling.top_p, sampling.top_k, sampling.temperature
        if 0 < top_p < 1.0:
            sl, _ = torch.sort(lp)
            sp = torch.exp(sl)
            above = torch.flip(torch.cumsum(torch.flip(sp, (0,)), 0), (0,)) - sp  # mass strictly above, exclusive
            total = above[:1] + sp[:1]
            n_drop = int((above >= top_p * total).sum())
            lp = torch.where(lp < sl[n_drop], torch.full_like(lp, -float("inf")), lp)
        if top_k > 0 and top_k < lp.shape[-1]:
            kth = torch.topk(lp, top_k).values[-1]
            lp = torch.where(lp < kth, torch.full_like(lp, -float("inf")), lp)
        probs = torch.softmax(lp / T, -1)
        return int(torch.multinomial(probs, 1, generator=gen))

    def decode_segments(self, ids, segments, draws=1, seeds=None, sampling=SERVED, abort_on_unclosed=True):
        torch = self.torch
        plan_segments(segments)
        ids = [int(t) for t in ids]
        if not ids:
            raise ValueError("empty prompt")
        seeds = self._seeds(draws, seeds)
        outs = []
        with torch.no_grad():
            cache = self._prefill(ids[:-1]) if len(ids) > 1 else None
            for d in range(draws):
                gen = torch.Generator(device=self.device).manual_seed(seeds[d])
                c = copy.deepcopy(cache)
                pos = len(ids) - 1
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
                    lg, c = self._forward(pending, c, pos, logits_to_keep=1)
                    pos += len(pending)
                    pending = []
                    logits = lg[0, -1]
                    out, closed, stop = [], 0, None
                    stops = set(seg.stop_ids)
                    for _ in range(seg.max_new):
                        t = int(logits.float().argmax()) if d == 0 else self._sample(logits, gen, sampling)
                        if t in stops:
                            closed, stop, last_stop = 1, t, t
                            break
                        out.append(t)
                        lg, c = self._forward([t], c, pos, logits_to_keep=1)
                        pos += 1
                        logits = lg[0, -1]
                    res.append(SegmentResult(tokens=out, closed=closed, stop_id=stop))
                    if not closed:
                        last_stop = None
                        if abort_on_unclosed:
                            aborted = True
                outs.append(res)
        return outs

    def logprobs(self, ids, positions) -> np.ndarray:
        torch = self.torch
        ids = [int(t) for t in ids]
        pos = self._positions(positions)
        p0, p1 = min(pos), max(pos)
        if p1 >= len(ids):
            raise ValueError("position beyond the sequence")
        with torch.no_grad():
            cache = self._prefill(ids[:p0]) if p0 else None
            lg, _ = self._forward(ids[p0:p1 + 1], cache, p0)
            lg = lg[0].float()
            lp = lg - torch.logsumexp(lg, -1, keepdim=True)
            return lp[torch.tensor([p - p0 for p in pos], device=lp.device)].cpu().numpy()

    # ---- manual forward with captures -----------------------------------------------------------------------
    def _attend(self, q, k, v, sliding: bool, window: int, scale: float):
        """q [1, H, L, D], k/v [1, KVH, L, D] -> [1, H, L, D]; causal (+ window), query-chunked."""
        torch = self.torch
        import torch.nn.functional as F

        H, L = q.shape[1], q.shape[2]
        rep = H // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        outs = []
        C = self.query_chunk
        for s in range(0, L, C):
            e = min(s + C, L)
            k0 = max(0, s - window + 1) if sliding else 0
            qi = torch.arange(s, e, device=q.device)[:, None]
            kj = torch.arange(k0, e, device=q.device)[None, :]
            allowed = (kj <= qi) & ((kj > qi - window) if sliding else torch.ones_like(kj, dtype=torch.bool))
            outs.append(F.scaled_dot_product_attention(q[:, :, s:e], k[:, :, k0:e], v[:, :, k0:e],
                                                       attn_mask=allowed[None, None], scale=scale))
        return torch.cat(outs, dim=2)

    def capture(self, ids, positions, layers=(), what=("resid",), attn_layers=(), patch_resid=None) -> dict:
        torch = self.torch
        from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

        what = check_capture_request(what)
        layers = sorted({int(i) for i in layers})
        attn_layers = sorted({int(i) for i in attn_layers}) if "attn" in what else []
        ids = [int(t) for t in ids]
        pos_l = self._positions(positions)
        if max(pos_l) >= len(ids):
            raise ValueError("position beyond the sequence")
        lm, cfg = self.lm, self.text_config
        n = len(lm.layers)
        for i in layers + attn_layers:
            if not 0 <= i < n:
                raise ValueError(f"layer {i} out of range")
        L = len(ids)
        W = cfg.sliding_window
        keep = {k: [] for k in ("resid", "resid_pre", "resid_mid", "mlp", "mlp_out", "pli", "attn")}
        f32 = lambda a: a.float().cpu().numpy()  # noqa: E731
        with torch.no_grad():
            x = torch.tensor([ids], device=self.device)
            emb = lm.embed_tokens(x)  # Gemma4TextScaledWordEmbedding: already times sqrt(hidden)
            h = emb
            pli_all = lm.project_per_layer_inputs(emb, lm.get_per_layer_inputs(x, emb)) if lm.hidden_size_per_layer_input else None
            position_ids = torch.arange(L, device=self.device)[None]
            pe = {t: lm.rotary_emb(h, position_ids, t) for t in set(cfg.layer_types)}
            pos = torch.tensor(pos_l, device=self.device)
            shared = {}
            for i, layer in enumerate(lm.layers):
                if getattr(layer, "enable_moe_block", False):
                    raise NotImplementedError("MoE layers are not supported")
                want = i in layers
                ltype = cfg.layer_types[i]
                cos, sin = pe[ltype]
                if want and "resid_pre" in what:
                    keep["resid_pre"].append(f32(h[0, pos]))
                residual = h
                xa = layer.input_layernorm(h)
                att = layer.self_attn
                hd = att.head_dim
                q = att.q_norm(att.q_proj(xa).view(1, L, -1, hd))
                q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2).transpose(1, 2)
                if att.is_kv_shared_layer:
                    k, v = shared[ltype]
                else:
                    k = att.k_proj(xa).view(1, L, -1, hd)
                    v = att.v_proj(xa).view(1, L, -1, hd) if att.v_proj is not None else k
                    k = apply_rotary_pos_emb(att.k_norm(k), cos, sin, unsqueeze_dim=2).transpose(1, 2)
                    v = att.v_norm(v).transpose(1, 2)
                if att.store_full_length_kv:
                    shared[ltype] = (k, v)
                if i in attn_layers:
                    kk = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1).float()
                    s = (q[:, :, pos].float() @ kk.transpose(-1, -2)) * att.scaling
                    j = torch.arange(L, device=self.device)[None, :]
                    p_ = pos[:, None]
                    allowed = (j <= p_) & ((j > p_ - W) if att.is_sliding else torch.ones_like(j, dtype=torch.bool))
                    s = torch.where(allowed[None, None], s, torch.full_like(s, -1e30))
                    keep["attn"].append(torch.softmax(s, -1)[0].cpu().numpy())
                o = self._attend(q, k, v, att.is_sliding, W, att.scaling)
                a_out = att.o_proj(o.transpose(1, 2).reshape(1, L, -1))
                h = residual + layer.post_attention_layernorm(a_out)
                if want and "resid_mid" in what:
                    keep["resid_mid"].append(f32(h[0, pos]))
                xm = layer.pre_feedforward_layernorm(h)
                mlp = layer.mlp
                act = mlp.act_fn(mlp.gate_proj(xm)) * mlp.up_proj(xm)
                if want and "mlp" in what:
                    keep["mlp"].append(f32(act[0, pos]))
                mo = mlp.down_proj(act)
                if want and "mlp_out" in what:
                    keep["mlp_out"].append(f32(mo[0, pos]))
                h = h + layer.post_feedforward_layernorm(mo)
                if layer.hidden_size_per_layer_input and pli_all is not None:
                    pli = pli_all[:, :, i, :]
                    if want and "pli" in what:
                        keep["pli"].append(f32(pli[0, pos]))
                    g = layer.act_fn(layer.per_layer_input_gate(h)) * pli
                    h = h + layer.post_per_layer_input_norm(layer.per_layer_projection(g))
                h = h * layer.layer_scalar
                if patch_resid and i in patch_resid:  # activation patching: overwrite the residual after layer i
                    h = h.clone()
                    for p_, v in patch_resid[i].items():
                        h[0, int(p_)] = torch.as_tensor(np.asarray(v), dtype=h.dtype, device=h.device)
                if want and "resid" in what:
                    keep["resid"].append(f32(h[0, pos]))
            out = {"positions": np.array(pos_l), "layers": np.array(layers, dtype=np.int64),
                   "attn_layers": np.array(attn_layers, dtype=np.int64)}
            for kname, v in keep.items():
                if v:
                    out[kname] = np.stack(v)
            if "logits" in what:
                hn = lm.norm(h[0, pos])
                lg = self.model.lm_head(hn).float()
                sc = cfg.final_logit_softcapping
                if sc:
                    lg = torch.tanh(lg / sc) * sc
                out["logits"] = lg.cpu().numpy()
        return out
