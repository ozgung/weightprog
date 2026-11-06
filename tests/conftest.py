"""Shared fixtures: one tiny random Gemma 4 checkpoint per session, loaded in float32 on CPU by each backend."""

from __future__ import annotations

import os

import pytest

from weightprog.backends import available, get_backend
from weightprog.testing import write_tiny_model

BACKENDS = available()

if "mlx" in BACKENDS:
    import mlx.core as mx

    mx.set_default_device(mx.cpu)  # tests never touch the GPU (it may be busy with real models)


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory):
    return write_tiny_model(tmp_path_factory.mktemp("tiny_g4"), seed=1)


@pytest.fixture(scope="session")
def tiny_dir_bf16(tmp_path_factory):
    return write_tiny_model(tmp_path_factory.mktemp("tiny_g4_bf16"), seed=1, dtype="BF16")


_loaded = {}


def _backend(name, d):
    if name not in BACKENDS:
        pytest.skip(f"{name} backend not installed")
    key = (name, str(d))
    if key not in _loaded:
        _loaded[key] = get_backend(name, d, dtype="float32", device="cpu")
    return _loaded[key]


@pytest.fixture(scope="session")
def mlx_be(tiny_dir):
    return _backend("mlx", tiny_dir)


@pytest.fixture(scope="session")
def torch_be(tiny_dir):
    return _backend("torch", tiny_dir)


@pytest.fixture(params=["mlx", "torch"], scope="session")
def be(request, tiny_dir):
    return _backend(request.param, tiny_dir)


@pytest.fixture
def sample_ids(tiny_dir):
    from weightprog.backends.base import Tokenizer

    tok = Tokenizer(tiny_dir)
    return tok.encode("class A:\n    def f(self, x):\n        return x * 2  # long enough to pass the window")


def env_path(var: str):
    p = os.environ.get(var)
    return p if p and os.path.exists(p) else None
