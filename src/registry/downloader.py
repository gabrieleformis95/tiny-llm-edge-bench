"""Download GGUFs from HuggingFace Hub to local cache."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

from src.config import settings
from src.registry.models import get_model, get_quant


def download_gguf(model_id: str, quant_name: str) -> Path:
    """Download a single GGUF to cache_dir. Idempotent.

    Returns path to the local file (cached or freshly downloaded).
    """
    model = get_model(model_id)
    quant = get_quant(quant_name)

    cache_dir = settings.gguf_cache_dir / model_id / quant_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    suffix = quant.file_suffix

    cached = next(
        (p for p in cache_dir.glob("*.gguf") if p.name.lower().endswith(suffix.lower())),
        None,
    )
    if cached:
        print(f"[cache] {cached}")
        return cached

    candidates = [
        f for f in list_repo_files(model.hf_gguf_repo)
        if f.endswith(suffix) or f.lower().endswith(suffix.lower())
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No GGUF file matching {suffix!r} in {model.hf_gguf_repo}"
        )
    filename = candidates[0]

    local_path = cache_dir / Path(filename).name

    print(f"[download] {model.hf_gguf_repo}/{filename} -> {local_path}")
    downloaded = hf_hub_download(
        repo_id=model.hf_gguf_repo,
        filename=filename,
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
        token=settings.hf_token or None,
    )
    return Path(downloaded)
