"""Model inputs for training: the serving code's ``main.ensemble.model_inputs`` plus a BGE embedding cache."""
from __future__ import annotations

import hashlib
import re
from importlib.metadata import version
from pathlib import Path

import numpy as np

from main.ensemble import EMBEDDING_DIM, model_inputs


def xgboost_features(texts: list[str]) -> np.ndarray:
    """(n, 525) float32 XGBoost inputs."""
    return model_inputs(texts)[0]


def neural_features(texts: list[str]) -> tuple[list[str], np.ndarray]:
    """(redacted texts, (n, 13) float32 numeric inputs) for the deep head."""
    _, redacted, numeric = model_inputs(texts)
    return redacted, numeric


def encoder_id() -> str:
    return "BAAI/bge-small-en-v1.5:fastembed-" + version("fastembed")   # main/retrieval/embeddings.py FastEmbedder.model_id


def embedding_key(texts: list[str], encoder: str) -> str:
    return hashlib.sha256(("\x00".join(texts) + "\x00" + encoder).encode("utf-8")).hexdigest()


def cached_embeddings(redacted_texts: list[str], cache_dir: Path, *, threads: int = 4, embedder=None) -> np.ndarray:
    """(n, 384) float32 BGE passage embeddings, cached as <cache_dir>/<encoder>_<sha16>.npz."""
    cache_dir = Path(cache_dir)
    encoder = embedder.model_id if embedder is not None else encoder_id()
    key = embedding_key(redacted_texts, encoder)
    path = cache_dir / f"{re.sub(r'[^A-Za-z0-9]+', '_', encoder)}_{key[:16]}.npz"
    if path.exists():
        with np.load(path) as archive:
            if str(archive["key"]) == key and archive["embeddings"].shape == (len(redacted_texts), EMBEDDING_DIM):
                return archive["embeddings"].astype(np.float32)
    if embedder is None:
        from main.ensemble import BGE_CACHE_DIR
        from main.retrieval.embeddings import FastEmbedder
        embedder = FastEmbedder(cache_dir=BGE_CACHE_DIR, threads=threads)
    embeddings = np.asarray(embedder.passages(redacted_texts), dtype=np.float32)
    if embeddings.shape != (len(redacted_texts), EMBEDDING_DIM) or not np.isfinite(embeddings).all():
        raise ValueError("Invalid BGE embeddings")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, embeddings=embeddings, key=np.asarray(key), encoder_id=np.asarray(encoder))
    return embeddings
