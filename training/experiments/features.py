"""Feature extraction copied from the shipped trainers (main.* only; fastembed loads lazily)."""
from __future__ import annotations

import hashlib
import re
from importlib.metadata import version
from pathlib import Path

import numpy as np

from main.domain.classification import MemoryClassifier
from main.domain.models import MemoryInput, MemoryRecord
from main.domain.privacy import redact_sensitive
from main.ml import FEATURE_NAMES, MLFeatureExtractor, TEXT_FEATURES

EMBEDDING_DIM = 384


def xgboost_features(texts: list[str]) -> np.ndarray:
    """(n, 525) float32; copy of training/train_conversational_retention.py:74-89 without the label loop."""
    classifier = MemoryClassifier(legacy=True)
    extractor = MLFeatureExtractor()
    matrix = []
    for text in texts:
        incoming = MemoryInput(content=text, user_id="dataset", session_id="dataset")
        record = MemoryRecord.from_input(incoming, classifier.classify(incoming))
        features = extractor.extract(incoming, record)
        matrix.append([features[name] for name in FEATURE_NAMES])
    return np.asarray(matrix, dtype=np.float32).reshape(len(texts), len(FEATURE_NAMES))


def neural_features(texts: list[str]) -> tuple[list[str], np.ndarray]:
    """(redacted texts, (n, 13) float32); copy of training/train_neural_retention.py:38-51."""
    classifier = MemoryClassifier(legacy=True)
    extractor = MLFeatureExtractor()
    redacted, numeric = [], []
    for raw in texts:
        text, _ = redact_sensitive(raw)
        incoming = MemoryInput(content=text, user_id="dataset", session_id="dataset")
        record = MemoryRecord.from_input(incoming, classifier.classify(incoming))
        features = extractor.extract(incoming, record)
        redacted.append(text)
        numeric.append([features[name] for name in TEXT_FEATURES])
    return redacted, np.asarray(numeric, dtype=np.float32).reshape(len(texts), len(TEXT_FEATURES))


def encoder_id() -> str:
    return "BAAI/bge-small-en-v1.5:fastembed-" + version("fastembed")   # main/retrieval/embeddings.py:36


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
        from main.retrieval.embeddings import FastEmbedder   # cache_dir='data/rag/models' is CWD-relative: run from ROOT
        embedder = FastEmbedder(threads=threads)
    embeddings = np.asarray(embedder.passages(redacted_texts), dtype=np.float32)
    if embeddings.shape != (len(redacted_texts), EMBEDDING_DIM) or not np.isfinite(embeddings).all():
        raise ValueError("Invalid BGE embeddings")   # training/train_neural_retention.py:53-54
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, embeddings=embeddings, key=np.asarray(key), encoder_id=np.asarray(encoder))
    return embeddings
