"""Memory Core orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .classification import MemoryClassifier
from .features import FeatureExtractor
from .lifecycle import LifecycleManager
from .models import MemoryInput, MemoryRecord
from .scoring import ImportanceScorer, ImportanceScoring
from .models import MemoryCategory
from .deadlines import parse_deadline
from .text_rules import recurring
from .segmentation import split_input
from .tasks import task_action
from .recurrence import parse_recurrence, scheduled_at
from .claims import extract_claim
from .reconciliation import validate_evidence


@dataclass(slots=True)
class MemoryCore:
    """Processes raw interactions into normalized memory records."""

    classifier: MemoryClassifier = field(default_factory=MemoryClassifier)
    feature_extractor: FeatureExtractor = field(default_factory=FeatureExtractor)
    importance_scorer: ImportanceScoring = field(default_factory=ImportanceScorer)
    lifecycle_manager: LifecycleManager = field(default_factory=LifecycleManager)

    @classmethod
    def with_ml(cls, model_dir: str | Path | None = None) -> "MemoryCore":
        """Replace heuristic scoring with the experimental Hippocorpus model."""
        from .ml import DEFAULT_MODEL_DIR, MLFeatureExtractor, MLImportanceScorer
        return cls(classifier=MemoryClassifier(legacy=True), feature_extractor=MLFeatureExtractor(),
                   importance_scorer=MLImportanceScorer(model_dir or DEFAULT_MODEL_DIR))

    def process(self, memory_input: MemoryInput) -> MemoryRecord:
        from .graph import validate_graph_metadata
        validate_graph_metadata(memory_input.metadata, memory_input.timestamp)
        category = self.classifier.classify(memory_input)
        if (not self.classifier.legacy and memory_input.role == 'user'
                and memory_input.metadata.get('relations') and category is MemoryCategory.TEMPORARY):
            category = MemoryCategory.SEMANTIC
        record = MemoryRecord.from_input(memory_input=memory_input, category=category)
        record.confidence = memory_input.metadata.get('confidence', record.confidence)
        validate_evidence(record)
        if record.role == 'user' and category in {MemoryCategory.SEMANTIC, MemoryCategory.PREFERENCE}:
            record.claim = extract_claim(record.content)
        action = task_action(record.content)
        is_reschedule = action is not None and action.kind == 'rescheduled'
        if memory_input.metadata.get("due_at") is not None and category is not MemoryCategory.TASK and not is_reschedule:
            raise ValueError("An explicit due_at requires a task message")
        if memory_input.metadata.get('due_at') is not None:
            parse_deadline('', memory_input.timestamp, memory_input.metadata['due_at'])
        if category is MemoryCategory.TASK:
            record.task_status = "active"
            if recurring(record.content):
                record.recurrence = parse_recurrence(record.content, memory_input.timestamp,
                                                     memory_input.metadata.get('due_at'))
                if record.recurrence:
                    record.next_due_at = scheduled_at(record.recurrence)
                else:
                    record.source_metadata['recurrence_resolution'] = 'unsupported_schedule'
            # Recurrence is a standing instruction; do not expire the entire series.
            if not recurring(record.content):
                record.due_at = parse_deadline(record.content, memory_input.timestamp,
                                               memory_input.metadata.get("due_at"))
        features = self.feature_extractor.extract(memory_input, record)
        score = self.importance_scorer.score(features)
        tier = self.lifecycle_manager.assign_tier(record, score)

        record.features = features
        record.importance_score = score
        record.tier = tier
        record.expires_at = self.lifecycle_manager.expiry_for(record)
        record.archive_after = self.lifecycle_manager.archive_after_for(record)
        from .ml import MLImportanceScorer
        if isinstance(self.importance_scorer, MLImportanceScorer):
            record.source_metadata["importance_model"] = {
                "type": "xgboost", "target": self.importance_scorer.metadata["target"],
                "sha256": self.importance_scorer.metadata["model_sha256"],
            }
        return record

    def process_many(self, memory_input: MemoryInput) -> list[MemoryRecord]:
        """Produce separate records for explicit independent clauses."""
        return [self.process(part) for part in split_input(memory_input)]
