"""Feature schema of the importance model: 13 text features plus 512 hashed word counts.

``main.ensemble.model_inputs`` builds the model's inputs with ``MLFeatureExtractor``; the scorer itself lives in
``main/ensemble.py``.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass

from .domain.features import FeatureExtractor
from .domain.models import MemoryInput, MemoryRecord

# Preserve the established feature schema identifier for artifact compatibility.
FEATURE_VERSION = "hippocorpus-text-v1"
TEXT_FEATURES = (
    "category_episodic", "category_preference", "category_procedural",
    "category_semantic", "category_task", "category_temporary", "entity_density",
    "has_deadline", "preference_signal", "sentiment_strength", "task_signal",
    "word_count_norm", "text_length_log",
)
FEATURE_NAMES = list(TEXT_FEATURES) + [f"lexical_{i}" for i in range(512)]


@dataclass
class MLFeatureExtractor(FeatureExtractor):
    """Add deterministic hashed word counts, with no fitted vocabulary or labels."""

    legacy: bool = True

    def extract(self, memory_input: MemoryInput, record: MemoryRecord) -> dict[str, float]:
        features = super().extract(memory_input, record)
        words = re.findall(r"\b[\w'-]+\b", memory_input.content.casefold())
        counts = Counter(int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") % 512 for word in words)
        features["text_length_log"] = math.log1p(len(words))
        features.update({f"lexical_{i}": math.log1p(counts.get(i, 0)) for i in range(512)})
        return features
