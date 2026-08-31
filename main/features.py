"""Feature extraction for memory scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

from .models import MemoryCategory, MemoryInput, MemoryRecord


POSITIVE_WORDS = {
    "amazing",
    "best",
    "enjoy",
    "favorite",
    "favourite",
    "good",
    "great",
    "happy",
    "like",
    "love",
    "prefer",
}

NEGATIVE_WORDS = {
    "angry",
    "bad",
    "dislike",
    "hate",
    "poor",
    "sad",
    "terrible",
    "upset",
    "worst",
}

ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


@dataclass(slots=True)
class FeatureExtractor:
    """Extracts simple numeric signals used by the baseline scorer."""

    def extract(self, memory_input: MemoryInput, record: MemoryRecord) -> dict[str, float]:
        text = memory_input.content
        lowered = text.lower()
        words = re.findall(r"\b[\w'-]+\b", lowered)
        word_count = len(words)
        entities = ENTITY_PATTERN.findall(text)

        positive_count = sum(1 for word in words if word in POSITIVE_WORDS)
        negative_count = sum(1 for word in words if word in NEGATIVE_WORDS)
        sentiment = self._sentiment(positive_count, negative_count)

        age_seconds = max(0.0, (self._now() - record.created_at).total_seconds())
        recency = max(0.0, 1.0 - min(age_seconds / 86_400.0, 1.0))
        entity_density = min(len(entities) / max(word_count, 1), 1.0)
        interaction_signal = min(float(memory_input.metadata.get("interaction_score", 0.0)), 1.0)

        return {
            "access_frequency": min(record.access_count / 10.0, 1.0),
            "category_episodic": self._is_category(record, MemoryCategory.EPISODIC),
            "category_preference": self._is_category(record, MemoryCategory.PREFERENCE),
            "category_procedural": self._is_category(record, MemoryCategory.PROCEDURAL),
            "category_semantic": self._is_category(record, MemoryCategory.SEMANTIC),
            "category_task": self._is_category(record, MemoryCategory.TASK),
            "category_temporary": self._is_category(record, MemoryCategory.TEMPORARY),
            "entity_density": entity_density,
            "has_deadline": self._bool_signal(r"\b(deadline|due|tomorrow|today|next week|schedule)\b", lowered),
            "interaction_signal": interaction_signal,
            "preference_signal": self._bool_signal(r"\b(like|love|prefer|favorite|favourite|dislike|hate)\b", lowered),
            "recency": recency,
            "sentiment_strength": abs(sentiment),
            "task_signal": self._bool_signal(r"\b(remind|todo|to-do|task|follow up|need to|please)\b", lowered),
            "word_count_norm": min(word_count / 80.0, 1.0),
        }

    @staticmethod
    def _bool_signal(pattern: str, text: str) -> float:
        return 1.0 if re.search(pattern, text, re.I) else 0.0

    @staticmethod
    def _is_category(record: MemoryRecord, category: MemoryCategory) -> float:
        return 1.0 if record.category is category else 0.0

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _sentiment(positive_count: int, negative_count: int) -> float:
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        return (positive_count - negative_count) / total
