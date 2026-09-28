"""Local cross-encoder reranking with a small injectable contract."""
from __future__ import annotations

import math
from typing import Iterable, Protocol


class Reranker(Protocol):
    model_id: str

    def rerank(self, query: str, documents: Iterable[str]) -> list[float]: ...


class FastEmbedCrossEncoderReranker:
    """Lazy local MS MARCO cross-encoder; memory text never leaves the machine."""

    model_id = 'Xenova/ms-marco-MiniLM-L-6-v2'

    def __init__(self, cache_dir='data/rag/models', threads=2):
        self.cache_dir = str(cache_dir)
        self.threads = threads
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError as exc:
                raise ValueError('Install fastembed to enable local reranking') from exc
            self._model = TextCrossEncoder(
                model_name=self.model_id, cache_dir=self.cache_dir, threads=self.threads)
        return self._model

    def rerank(self, query: str, documents: Iterable[str]) -> list[float]:
        documents = list(documents)
        scores = [float(value) for value in self.model.rerank(query, documents, batch_size=64)]
        if len(scores) != len(documents) or any(not math.isfinite(value) for value in scores):
            raise ValueError('Reranker returned invalid scores')
        return scores
