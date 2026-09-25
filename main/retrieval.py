"""Deterministic ranking for local keyword retrieval."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

from .models import MemoryCategory, MemoryRecord, MemoryTier, utc_now


def tokenize(text: str) -> set[str]:
    """Use unique, case-insensitive whole words; ignore punctuation."""
    return set(re.findall(r"\w+", text.casefold()))


@dataclass(slots=True)
class MemoryRanker:
    """Rank lexical matches using bounded signals and configurable weights.

    Category and tier scores are usefulness priors, not inferred query intent.
    Recency uses creation time with a 30-day half-life by default.
    """

    weights: dict[str, float] = field(default_factory=lambda: {
        "keyword_relevance": 0.65,
        "category": 0.10,
        "tier": 0.05,
        "recency": 0.10,
        "importance": 0.10,
    })
    recency_half_life_days: float = 30.0

    def __post_init__(self) -> None:
        names = {"keyword_relevance", "category", "tier", "recency", "importance"}
        if set(self.weights) != names:
            raise ValueError("Ranking weights must specify all five ranking signals")
        if any(not math.isfinite(value) or value < 0 for value in self.weights.values()):
            raise ValueError("Ranking weights must be finite and nonnegative")
        if not math.isfinite(sum(self.weights.values())) or sum(self.weights.values()) <= 0:
            raise ValueError("Ranking weights must have a finite positive total")
        if not math.isfinite(self.recency_half_life_days) or self.recency_half_life_days <= 0:
            raise ValueError("Recency half-life must be finite and positive")

    def rank(
        self,
        query: str,
        records: Iterable[MemoryRecord],
        *,
        now: datetime | None = None,
    ) -> list[MemoryRecord]:
        terms = tokenize(query)
        if not terms:
            return []
        now = now or utc_now()
        category_scores = {
            MemoryCategory.SEMANTIC: 1.0,
            MemoryCategory.PREFERENCE: 1.0,
            MemoryCategory.PROCEDURAL: 0.9,
            MemoryCategory.TASK: 0.9,
            MemoryCategory.EPISODIC: 0.6,
            MemoryCategory.TEMPORARY: 0.1,
        }
        tier_scores = {
            MemoryTier.LONG_TERM: 1.0,
            MemoryTier.SHORT_TERM: 0.7,
            MemoryTier.WORKING: 0.4,
            MemoryTier.ARCHIVE: 0.2,
        }
        scored = []
        for record in records:
            matches = terms & tokenize(record.content)
            if not matches:
                continue
            age_days = max(0.0, (now - record.created_at).total_seconds() / 86400)
            importance = record.importance_score
            signals = {
                "keyword_relevance": len(matches) / len(terms),
                "category": category_scores[record.category],
                "tier": tier_scores[record.tier],
                "recency": 2 ** (-age_days / self.recency_half_life_days),
                "importance": max(0.0, min(importance, 1.0)) if math.isfinite(importance) else 0.0,
            }
            score = sum(self.weights[name] * value for name, value in signals.items())
            score /= sum(self.weights.values())
            scored.append((score, record))

        # Stable across store order: score descending, newest first, then ID.
        scored.sort(key=lambda item: (-item[0], -item[1].created_at.timestamp(), item[1].id))
        return [record for _, record in scored]
