"""Local neural embeddings; no network call is made with memory text."""
from __future__ import annotations

import math
from typing import Protocol


class Embedder(Protocol):
    model_id: str
    dimension: int

    def passages(self, texts: list[str]) -> list[list[float]]: ...
    def query(self, text: str) -> list[float]: ...


def normalized(vector, dimension: int) -> list[float]:
    result = [float(value) for value in vector]
    if len(result) != dimension or any(not math.isfinite(value) for value in result):
        raise ValueError('Embedding has an invalid dimension or nonfinite values')
    norm = math.hypot(*result)
    if norm == 0 or not math.isfinite(norm):
        raise ValueError('Embedding must have a finite nonzero norm')
    return [value / norm for value in result]


class FastEmbedder:
    """BGE neural embeddings with separately encoded queries and passages."""
    dimension = 384

    def __init__(self, cache_dir='data/rag/models', threads=2, local_files_only=False):
        try:
            from fastembed import TextEmbedding
            from importlib.metadata import version
        except ImportError as exc:
            raise ValueError('Install requirements.txt to enable semantic retrieval') from exc
        self.model_id = 'BAAI/bge-small-en-v1.5:fastembed-' + version('fastembed')
        # local_files_only: fail instead of downloading when the model is not cached (the importance scorer).
        self.model = TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir=str(cache_dir), threads=threads,
                                   local_files_only=local_files_only)

    def passages(self, texts: list[str]) -> list[list[float]]:
        return [normalized(vector, self.dimension) for vector in self.model.passage_embed(texts, batch_size=32)]

    def queries(self, texts: list[str]) -> list[list[float]]:
        return [normalized(vector, self.dimension) for vector in self.model.query_embed(texts, batch_size=32)]

    def query(self, text: str) -> list[float]:
        return self.queries([text])[0]
