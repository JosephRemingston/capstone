"""Memory Core orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..domain.classification import MemoryClassifier
from ..domain.features import FeatureExtractor
from ..domain.lifecycle import LifecycleManager
from ..domain.models import MemoryInput, MemoryRecord
from ..domain.scoring import ImportanceScorer, ImportanceScoring
from ..domain.models import MemoryCategory
from ..domain.deadlines import parse_deadline
from ..domain.text_rules import recurring
from ..domain.segmentation import split_input
from ..domain.tasks import task_action
from ..domain.recurrence import parse_recurrence, scheduled_at
from ..domain.claims import extract_claim
from ..domain.reconciliation import validate_evidence
from ..domain.privacy import redact_metadata, redact_sensitive
from ..ensemble import default_importance_scorer


@dataclass(slots=True)
class MemoryCore:
    """Processes raw interactions into normalized memory records."""

    classifier: MemoryClassifier = field(default_factory=MemoryClassifier)
    feature_extractor: FeatureExtractor = field(default_factory=FeatureExtractor)
    # Default: the hybrid scorer (25 % csv5k ensemble + 75 % heuristic), or COGNIMEM_SCORER=heuristic.
    importance_scorer: ImportanceScoring = field(default_factory=default_importance_scorer)
    lifecycle_manager: LifecycleManager = field(default_factory=LifecycleManager)

    @classmethod
    def with_heuristic(cls) -> "MemoryCore":
        """Use only the rule-based importance score (the COGNIMEM_SCORER=heuristic opt-out)."""
        return cls(importance_scorer=ImportanceScorer())

    def process(self, memory_input: MemoryInput) -> MemoryRecord:
        sanitized, sensitive_types = redact_sensitive(memory_input.content)
        sanitized_metadata, metadata_types = redact_metadata(memory_input.metadata)
        sensitive_types = list(dict.fromkeys([*sensitive_types, *metadata_types]))
        if sensitive_types:
            memory_input = replace(memory_input, content=sanitized, metadata={
                **sanitized_metadata,
                'privacy': {'redacted': True, 'types': sensitive_types},
            })
        elif sanitized_metadata != memory_input.metadata:
            memory_input = replace(memory_input, metadata=sanitized_metadata)
        from ..graph import validate_graph_metadata
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
        assess = getattr(self.importance_scorer, "assess", None)
        if callable(assess):
            score, provenance = assess(memory_input.content, features)
        else:
            text_scorer = getattr(self.importance_scorer, "score_text", None)
            score = (text_scorer(memory_input.content, features) if callable(text_scorer)
                     else self.importance_scorer.score(features))
            provenance = {"type": "heuristic" if isinstance(self.importance_scorer, ImportanceScorer)
                          else type(self.importance_scorer).__name__}
        tier = self.lifecycle_manager.assign_tier(record, score)

        record.features = features
        record.importance_score = score
        record.tier = tier
        record.expires_at = self.lifecycle_manager.expiry_for(record)
        record.archive_after = self.lifecycle_manager.archive_after_for(record)
        # Always overwritten, so a caller-supplied value cannot misstate which scorer ran.
        record.source_metadata["importance_model"] = provenance
        return record

    def process_many(self, memory_input: MemoryInput) -> list[MemoryRecord]:
        """Produce separate records for explicit independent clauses."""
        return [self.process(part) for part in split_input(memory_input)]
