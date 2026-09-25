"""Baseline heuristic memory importance scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class ImportanceScoring(Protocol):
    """Shared contract for heuristic and trained importance scorers."""

    def score(self, features: dict[str, float]) -> float:
        ...


@dataclass(slots=True)
class ImportanceScorer:
    """Scores extracted features on a 0.0 to 1.0 scale.

    The class intentionally exposes a tiny feature-dict interface so a trained
    LightGBM/XGBoost model can replace this baseline later.
    """

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "access_frequency": 0.08,
            "category_episodic": 0.20,
            "category_preference": 0.38,
            "category_procedural": 0.34,
            "category_semantic": 0.38,
            "category_task": 0.38,
            "entity_density": 0.08,
            "has_deadline": 0.09,
            "interaction_signal": 0.08,
            "preference_signal": 0.12,
            "recency": 0.08,
            "sentiment_strength": 0.05,
            "task_signal": 0.12,
            "word_count_norm": 0.06,
        }
    )
    temporary_penalty: float = 0.25

    def score(self, features: dict[str, float]) -> float:
        score = sum(self.weights.get(name, 0.0) * value for name, value in features.items())
        if features.get("category_temporary", 0.0) >= 1.0:
            score *= self.temporary_penalty

        return max(0.0, min(round(score, 4), 1.0))
