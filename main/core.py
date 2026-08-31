"""Memory Core orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

from .classification import MemoryClassifier
from .features import FeatureExtractor
from .lifecycle import LifecycleManager
from .models import MemoryInput, MemoryRecord
from .scoring import ImportanceScorer


@dataclass(slots=True)
class MemoryCore:
    """Processes raw interactions into normalized memory records."""

    classifier: MemoryClassifier = field(default_factory=MemoryClassifier)
    feature_extractor: FeatureExtractor = field(default_factory=FeatureExtractor)
    importance_scorer: ImportanceScorer = field(default_factory=ImportanceScorer)
    lifecycle_manager: LifecycleManager = field(default_factory=LifecycleManager)

    def process(self, memory_input: MemoryInput) -> MemoryRecord:
        category = self.classifier.classify(memory_input)
        record = MemoryRecord.from_input(memory_input=memory_input, category=category)
        features = self.feature_extractor.extract(memory_input, record)
        score = self.importance_scorer.score(features)
        tier = self.lifecycle_manager.assign_tier(record, score)

        record.features = features
        record.importance_score = score
        record.tier = tier
        record.expires_at = self.lifecycle_manager.expiry_for(record)
        record.archive_after = self.lifecycle_manager.archive_after_for(record)
        return record
