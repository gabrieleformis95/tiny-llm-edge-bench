"""Tests for AWQ backend - import path and platform error behavior."""

from __future__ import annotations

import pytest


def test_awq_backend_importable():
    from src.inference.awq_backend import AWQBackend, PlatformError
    assert AWQBackend is not None
    assert PlatformError is not None


def test_awq_backend_implements_interface():
    from src.inference.awq_backend import AWQBackend
    from src.inference.base import InferenceBackend
    # AWQBackend must satisfy the InferenceBackend protocol structurally
    assert hasattr(AWQBackend, "load")
    assert hasattr(AWQBackend, "generate")
    assert hasattr(AWQBackend, "unload")


def test_awq_backend_load_raises_platform_error_without_cuda(tmp_path):
    """On Apple Silicon / CPU-only, load() must raise PlatformError, not crash."""
    from src.inference.awq_backend import AWQBackend, PlatformError
    backend = AWQBackend()
    fake_path = tmp_path / "fake_awq_model"
    fake_path.mkdir()
    with pytest.raises((PlatformError, ImportError)):
        backend.load(fake_path)


def test_awq_backend_unload_before_load():
    from src.inference.awq_backend import AWQBackend
    backend = AWQBackend()
    backend.unload()  # must not raise
    assert backend._model is None
