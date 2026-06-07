"""Load and validate model/quant registry from configs/models.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from src.config import ModelSpec, QuantSpec

_MODELS_YAML = Path(__file__).parents[2] / "configs" / "models.yaml"
_QUANTS_YAML = Path(__file__).parents[2] / "configs" / "quants.yaml"


@lru_cache(maxsize=1)
def load_models() -> list[ModelSpec]:
    """Return all ModelSpec entries from models.yaml."""
    with open(_MODELS_YAML) as f:
        return [ModelSpec(**m) for m in yaml.safe_load(f)["models"]]


@lru_cache(maxsize=1)
def load_quants() -> list[QuantSpec]:
    """Return all QuantSpec entries from quants.yaml."""
    with open(_QUANTS_YAML) as f:
        return [QuantSpec(**q) for q in yaml.safe_load(f)["quants"]]


def get_model(model_id: str) -> ModelSpec:
    """Return a single ModelSpec by id. Raises KeyError if not found."""
    for m in load_models():
        if m.id == model_id:
            return m
    raise KeyError(f"Model not found in registry: {model_id!r}")


def get_quant(quant_name: str) -> QuantSpec:
    """Return a single QuantSpec by name. Raises KeyError if not found."""
    for q in load_quants():
        if q.name == quant_name:
            return q
    raise KeyError(f"Quant not found in registry: {quant_name!r}")
