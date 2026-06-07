"""Tests for inference backends (mocked - no real model loaded)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.inference.base import GenerationResult, InferenceBackend
from src.inference.llama_cpp_backend import LlamaCppBackend


def test_generation_result_fields():
    r = GenerationResult(
        text="Hello",
        ttft_ms=12.3,
        tpot_ms=4.5,
        total_tokens=10,
        prompt_tokens=5,
    )
    assert r.text == "Hello"
    assert r.total_tokens == 10


def test_inference_backend_is_abstract():
    class Incomplete(InferenceBackend):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_llama_cpp_backend_load_mocked():
    mock_llama_cls = MagicMock()
    with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_cls)}):
        backend = LlamaCppBackend()
        backend.load(Path("fake.gguf"), n_ctx=512, n_gpu_layers=0)
        mock_llama_cls.assert_called_once()


def test_llama_cpp_backend_generate_mocked():
    backend = LlamaCppBackend()
    mock_model = MagicMock()
    mock_model.return_value = {
        "choices": [{"text": "Paris"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
    }
    backend._model = mock_model

    result = backend.generate("What is the capital of France?", max_tokens=32)

    assert result.text == "Paris"
    assert result.prompt_tokens == 5


def test_llama_cpp_backend_unload():
    backend = LlamaCppBackend()
    backend._model = MagicMock()
    backend.unload()
    assert backend._model is None


def test_mlx_backend_stub():
    pytest.skip("Apple Silicon Metal only - not tested in CI")
