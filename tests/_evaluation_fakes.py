"""Deterministic stand-ins for the embedding and reranking models used by evaluation tests."""
import hashlib
import math
import re


class HashEmbedder:
    """Bag-of-words hashed into a small dense vector; good enough for ranking tests."""
    model_id = 'test-hash-embedder'
    dimension = 64

    def _embed(self, text):
        vector = [0.0] * self.dimension
        for token in re.findall(r'\w+', text.casefold()):
            vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def passages(self, texts):
        return [self._embed(text) for text in texts]

    def queries(self, texts):
        return [self._embed(text) for text in texts]

    def query(self, text):
        return self._embed(text)


class OverlapReranker:
    """Scores a document by query-token overlap, mimicking a cross-encoder's interface."""
    model_id = 'test-overlap-reranker'

    def rerank(self, query, documents):
        terms = set(re.findall(r'\w+', query.casefold()))
        return [float(len(terms & set(re.findall(r'\w+', doc.casefold())))) for doc in documents]
