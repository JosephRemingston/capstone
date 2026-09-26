"""Fact/preference reconciliation, supported by preserved source observations."""
from dataclasses import replace
from datetime import datetime
import math
import re

from .claims import claim_summary, contradicts, equivalent, extract_claim, normalized_value
from .lifecycle import LifecycleManager
from .models import MemoryCategory, MemoryRecord, MemoryTier
from .text_rules import uncertain

_CATEGORIES = {MemoryCategory.SEMANTIC, MemoryCategory.PREFERENCE, MemoryCategory.PROCEDURAL}
_RANK_NAMES = ('source_priority', 'confidence', 'observation_time', 'importance')


def validate_evidence(record: MemoryRecord) -> None:
    for name, value, maximum in (
        ('confidence', record.confidence, 1),
        ('source_priority', record.source_metadata.get('source_priority', 0), 100),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f'{name} must be a finite number between 0 and {maximum}')


def eligible(record: MemoryRecord) -> bool:
    return (record.role == 'user' and record.category in _CATEGORIES and not uncertain(record.content)
            and not re.search(r'\b(?:used to|previously|formerly|yesterday|tomorrow|today|'
                              r'(?:last|next)\s+(?:week|month|year))\b', record.content, re.I))


def _prepared(record: MemoryRecord) -> MemoryRecord:
    # Derive assertions from original content, including legacy rows without fields.
    return replace(record, claim=extract_claim(record.content),
                   conflict_ids=list(record.conflict_ids), evidence_ids=list(record.evidence_ids),
                   source_metadata=dict(record.source_metadata))


def _same(left: MemoryRecord, right: MemoryRecord) -> bool:
    if left.claim and right.claim:
        return equivalent(left.claim, right.claim)
    return left.category == right.category and normalized_value(left.content) == normalized_value(right.content)


def _rank(record: MemoryRecord) -> tuple:
    validate_evidence(record)
    importance = min(1, max(0, record.importance_score)) if math.isfinite(record.importance_score) else 0
    return (record.source_metadata.get('source_priority', 0), record.confidence,
            record.last_confirmed_at or record.created_at, importance)


def _support_rank(record: MemoryRecord, sources: dict[str, MemoryRecord]) -> tuple:
    # Never combine the confidence of one source with the date of another.
    return max(_rank(item) for item in _supporting_sources(record, sources))


def _supporting_sources(record: MemoryRecord, sources: dict[str, MemoryRecord]) -> list[MemoryRecord]:
    identifiers = record.evidence_ids or [record.id]
    if record.id not in identifiers:
        raise ValueError('Evidence must include the original memory')
    evidence = []
    for identifier in identifiers:
        source = sources.get(identifier)
        if source is None or source.user_id != record.user_id or not eligible(source) or not _same(_prepared(source), record):
            raise ValueError('Evidence must exist, belong to the same user, and support the same assertion')
        evidence.append(source)
    return evidence


def _reason(winner_rank: tuple, loser_rank: tuple) -> str:
    for name, winner, loser in zip(_RANK_NAMES, winner_rank, loser_rank):
        if winner != loser:
            return name
    return 'tie_kept_existing'


def _refresh(record: MemoryRecord, when: datetime) -> None:
    manager = LifecycleManager()
    record.tier = MemoryTier.SHORT_TERM  # A fresh assertion can reactivate an archive view.
    record.tier = manager.assign_tier(record, record.importance_score, now=when)
    record.expires_at = manager.expiry_for(record)
    record.archive_after = manager.archive_after_for(record)


def reconcile(incoming: MemoryRecord, stored: list[MemoryRecord], *, preserve_evidence: bool = False) -> list[MemoryRecord]:
    """Annotate the incoming observation and return revisions to existing records.

    Only active roots participate. Reasserting an old value after a change starts
    a new episode rather than merging evidence across contradictory periods.
    """
    validate_evidence(incoming)
    if not eligible(incoming):
        return []
    incoming.claim = extract_claim(incoming.content)
    if not preserve_evidence or not incoming.evidence_ids:
        incoming.evidence_ids = [incoming.id]
        incoming.evidence_session_ids = [incoming.session_id]
        incoming.first_observed_at = incoming.last_observed_at = incoming.created_at
    candidates = [_prepared(item) for item in stored if item.user_id == incoming.user_id
                  and item.memory_status == 'active' and eligible(item)]
    sources = {item.id: item for item in [*stored, incoming]}
    revisions = {}
    duplicates = [item for item in candidates if _same(item, incoming)]
    canonical = incoming
    if duplicates:
        canonical = min(duplicates, key=lambda item: (item.created_at, item.id))
        evidence = set(incoming.evidence_ids)
        for item in duplicates:
            evidence.update(item.evidence_ids or [item.id])
        # Imported provenance cannot borrow support from another user/assertion.
        for item in [*duplicates, incoming]:
            _supporting_sources(item, sources)
        canonical.evidence_ids = sorted(evidence, key=lambda identifier: (sources[identifier].created_at, identifier))
        canonical.evidence_session_ids = sorted({sources[key].session_id for key in evidence})
        canonical.first_observed_at = min(sources[key].created_at for key in evidence)
        canonical.last_observed_at = max([sources[key].created_at for key in evidence] +
                                        [item.last_observed_at or item.created_at for item in duplicates])
        canonical.updated_at = max(canonical.updated_at, incoming.created_at)
        canonical.summary = claim_summary(canonical.claim) if canonical.claim else canonical.content
        _refresh(canonical, canonical.last_observed_at)
        revisions[canonical.id] = canonical
        for item in [*duplicates, incoming]:
            if item.id == canonical.id:
                continue
            item.memory_status = 'consolidated'
            item.consolidated_into = canonical.id
            item.updated_at = max(item.updated_at, incoming.created_at)
            if item.id != incoming.id:
                revisions[item.id] = item

    opponents = [item for item in candidates if canonical.claim and item.claim
                 and contradicts(canonical.claim, item.claim)]
    if opponents:
        canonical_rank = _support_rank(canonical, sources)
        # An equal-rank tie keeps the existing assertion, with a recorded reason.
        best_opponent = max(opponents, key=lambda item: (_support_rank(item, sources), item.id))
        opponent_rank = _support_rank(best_opponent, sources)
        wins = canonical_rank > opponent_rank
        winner = canonical if wins else best_opponent
        reason = _reason(canonical_rank, opponent_rank) if wins else _reason(opponent_rank, canonical_rank)
        decision = {'winner_id': winner.id, 'reason': reason, 'method': 'automatic',
                    'observation_id': incoming.id, 'recorded_at': incoming.created_at.isoformat()}
        canonical.conflict_ids = sorted(set(canonical.conflict_ids) | {item.id for item in opponents})
        canonical.conflict_resolution = decision
        for opponent in opponents:
            opponent.conflict_ids = sorted(set(opponent.conflict_ids) | {canonical.id})
            opponent.conflict_resolution = {**decision, 'decision': 'superseded' if wins else 'kept'}
            opponent.updated_at = max(opponent.updated_at, incoming.created_at)
            if wins:
                opponent.memory_status = 'superseded'
                opponent.superseded_by = canonical.id
            revisions[opponent.id] = opponent
        if not wins:
            canonical.memory_status = 'superseded'
            canonical.superseded_by = winner.id
        canonical.conflict_resolution = {**decision, 'decision': 'kept' if wins else 'superseded'}
        if canonical.id != incoming.id:
            revisions[canonical.id] = canonical
            incoming.conflict_resolution = canonical.conflict_resolution
            incoming.conflict_ids = list(canonical.conflict_ids)
    return list(revisions.values())


def select_current(selected: MemoryRecord, stored: list[MemoryRecord], when: datetime) -> list[MemoryRecord]:
    """Explicitly restore/select a root and supersede its active contradictions."""
    if not eligible(selected) or selected.memory_status == 'consolidated':
        raise ValueError('Select an original fact/preference root, not a consolidated observation')
    selected = _prepared(selected)
    if not selected.claim:
        raise ValueError('Explicit conflict resolution requires a supported assertion')
    opponents = [_prepared(item) for item in stored if item.id != selected.id
                 and item.user_id == selected.user_id and item.memory_status == 'active' and eligible(item)]
    opponents = [item for item in opponents if item.claim and contradicts(selected.claim, item.claim)]
    if not opponents:
        raise ValueError('No active contradictory memory to resolve')
    if when.tzinfo is None or any(when < item.updated_at for item in [selected, *opponents]):
        raise ValueError('Resolution time must be timezone-aware and not precede affected revisions')
    selected.memory_status = 'active'
    selected.superseded_by = None
    selected.updated_at = when
    selected.first_observed_at = selected.first_observed_at or selected.created_at
    selected.last_observed_at = when  # Explicit selection reaffirms the assertion now.
    selected.last_confirmed_at = when
    _refresh(selected, when)
    selected.conflict_ids = sorted(set(selected.conflict_ids) | {item.id for item in opponents})
    decision = {'winner_id': selected.id, 'reason': 'explicit_selection', 'method': 'manual',
                'recorded_at': when.isoformat()}
    selected.conflict_resolution = {**decision, 'decision': 'kept'}
    for item in opponents:
        item.memory_status = 'superseded'
        item.superseded_by = selected.id
        item.updated_at = when
        item.conflict_ids = sorted(set(item.conflict_ids) | {selected.id})
        item.conflict_resolution = {**decision, 'decision': 'superseded'}
    return [*opponents, selected]
