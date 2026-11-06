"""Backends: one interface (base.Backend) over MLX and PyTorch.

    from weightprog.backends import get_backend
    be = get_backend("mlx", model_dir, dtype="bfloat16")      # or "torch" (device="cuda" | "cpu")
"""

from __future__ import annotations

from .base import CAPTURE_KINDS, SERVED, Backend, Force, Generate, Sampling, SegmentResult, Tokenizer, index_seed, seed_of


def get_backend(name: str, model_dir=None, dtype: str | None = "bfloat16", device: str | None = None, **kw):
    """Instantiate a backend by name; loads the model when model_dir is given."""
    if name == "mlx":
        from .mlx_backend import MLXBackend
        return MLXBackend(model_dir, dtype=dtype, device=device, **kw)
    if name == "torch":
        from .torch_backend import TorchBackend
        return TorchBackend(model_dir, dtype=dtype, device=device, **kw)
    raise ValueError(f"unknown backend {name!r} (mlx | torch)")


def available() -> list[str]:
    out = []
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
        out.append("mlx")
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        out.append("torch")
    except Exception:  # noqa: BLE001
        pass
    return out


__all__ = ["Backend", "CAPTURE_KINDS", "Force", "Generate", "SERVED", "Sampling", "SegmentResult", "Tokenizer",
           "available", "get_backend", "index_seed", "seed_of"]
