"""Local JSONL-backed memory store."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ..domain.models import MemoryCategory, MemoryRecord, MemoryTier, utc_now
from ..retrieval import MemoryRanker, tokenize
from ..domain.lifecycle import LifecycleManager
from ..domain.tasks import task_action, match_tasks
from ..domain.task_updates import revise_task
from ..domain.deadlines import parse_deadline
from ..domain.reconciliation import reconcile, select_current, eligible
from .locking import locked


DEFAULT_STORE_PATH = Path("memory_store") / "memories.jsonl"


@dataclass(slots=True)
class LocalMemoryStore:
    """Append-only local storage for serialized memory records."""

    path: Path | str = DEFAULT_STORE_PATH
    ranker: MemoryRanker = field(default_factory=MemoryRanker)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def save(self, record: MemoryRecord) -> MemoryRecord:
        self._append([record])
        return record

    @locked
    def _append(self, records: list[MemoryRecord]) -> None:
        from ..graph import validate_graph_metadata
        for record in records:
            validate_graph_metadata(record.source_metadata, record.created_at)
        owners = {item.id: item.user_id for item in self.all()}
        for record in records:
            if record.id in owners and owners[record.id] != record.user_id:
                raise ValueError('A memory ID cannot change owner')
            owners[record.id] = record.user_id
        recorded_at = utc_now()
        for record in records:
            record.recorded_at = recorded_at
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(''.join(json.dumps(record.to_dict(), sort_keys=True) + '\n' for record in records))

    def all(self) -> list[MemoryRecord]:
        records = {record.id: record for record in self._read_log()}
        return list(records.values())

    @locked
    def _read_log(self) -> list[MemoryRecord]:
        if not self.path.exists():
            return []

        records: dict[str, MemoryRecord] = {}
        revisions = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                    record = MemoryRecord.from_dict(payload)
                    if record.id in records and records[record.id].user_id != record.user_id:
                        raise ValueError("A memory ID cannot change owner")
                    records[record.id] = record
                    revisions.append(record)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid memory record on line {line_number} in {self.path}") from exc
        return revisions

    def history(self, record_id: str, *, user_id: str) -> list[MemoryRecord]:
        """Return every persisted revision for this user's record in log order."""
        return [record for record in self._read_log() if record.id == record_id and record.user_id == user_id]

    @locked
    def resolve_conflict(self, record_id: str, *, user_id: str, now: datetime | None = None) -> MemoryRecord:
        records = self.all()
        selected = next((item for item in records if item.id == record_id and item.user_id == user_id), None)
        if selected is None:
            raise ValueError('Memory not found for this user')
        revisions = select_current(selected, records, now or utc_now())
        self._append(revisions)
        return revisions[-1]

    @locked
    def reconcile_memories(self, *, user_id: str) -> list[MemoryRecord]:
        """Reconcile existing active facts/preferences, appending only changed rows."""
        records = self.all()
        roots = sorted((item for item in records if item.user_id == user_id
                        and item.memory_status == 'active' and eligible(item)),
                       key=lambda item: (item.created_at, item.id))
        root_ids = {item.id for item in roots}
        context = {item.id: replace(item, memory_status='pending') if item.id in root_ids else item
                   for item in records}
        for original in roots:
            current = replace(original)
            revisions = reconcile(current, list(context.values()), preserve_evidence=True)
            context.update({item.id: item for item in [*revisions, current]})
        changed = [context[item.id] for item in records if context[item.id].to_dict() != item.to_dict()]
        if changed:
            self._append(changed)
        return changed

    @locked
    def ingest(self, record: MemoryRecord) -> MemoryRecord:
        """Persist an observation, reconcile facts/preferences, and apply task updates.

        Use save() for raw persistence. No access counts change during reads.
        """
        action = task_action(record.content) if record.role == 'user' else None
        target_id = record.source_metadata.get('task_id')
        if target_id is not None and (not isinstance(target_id, str) or not target_id or not action):
            raise ValueError('task_id requires a recognized task update by the user')
        scope = record.source_metadata.get('task_scope')
        if scope is not None and (scope not in ('occurrence', 'series') or not action):
            raise ValueError('task_scope requires a task update and must be occurrence or series')
        if 'occurrence_at' in record.source_metadata and not action:
            raise ValueError('occurrence_at requires a task update')
        if record.source_metadata.get('occurrence_at') is not None:
            parse_deadline('', record.created_at, record.source_metadata['occurrence_at'])
        records = self.all()
        existing = next((item for item in records if item.id == record.id), None)
        if existing:
            if existing.user_id != record.user_id or existing.content != record.content:
                raise ValueError('An ingested memory ID cannot be reused for a different observation')
            return existing  # Retrying the same event must not advance a series twice.
        updates = []
        if action:
            candidates = [item for item in records
                          if item.category is MemoryCategory.TASK and item.user_id == record.user_id
                          and item.task_status in (None, 'active') and item.updated_at <= record.created_at]
            if target_id:
                candidates = [item for item in candidates if item.id == target_id]
                if not candidates:
                    raise ValueError('Target task not found, inactive, newer than update, or belongs to another user')
            else:
                candidates = match_tasks(candidates, action, record.session_id)
            if len(candidates) == 1:
                target = candidates[0]
                revision = revise_task(target, record, action)
                if revision:
                    updates.append(revision)
            elif len(candidates) > 1:
                record.source_metadata['task_resolution'] = 'ambiguous'
                record.source_metadata['candidate_task_ids'] = [item.id for item in candidates]
            else:
                record.source_metadata['task_resolution'] = 'unmatched'
        updates.extend(reconcile(record, records))
        self._append(updates + [record])
        return record

    def _visible(self, *, include_expired: bool, include_resolved: bool, include_history: bool,
                 now: datetime | None) -> list[MemoryRecord]:
        when = now or utc_now()
        manager = LifecycleManager()
        return [manager.refresh(record, now=when) for record in self.all()
                if (include_expired or record.expires_at is None or record.expires_at > when)
                and (include_history or record.memory_status == 'active')
                and (include_resolved or record.task_status not in {'completed', 'cancelled'})]

    def get(self, record_id: str) -> MemoryRecord | None:
        for record in self.all():
            if record.id == record_id:
                return record
        return None

    def list(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        category: MemoryCategory | str | None = None,
        tier: MemoryTier | str | None = None,
        limit: int | None = None,
        include_expired: bool = False,
        include_resolved: bool = False,
        include_history: bool = False,
        now: datetime | None = None,
    ) -> list[MemoryRecord]:
        if limit is not None and limit < 0:
            raise ValueError('List limit must be nonnegative')
        records = self._filter(
            self._visible(include_expired=include_expired, include_resolved=include_resolved, include_history=include_history, now=now),
            user_id=user_id,
            session_id=session_id,
            category=category,
            tier=tier,
        )
        records.sort(key=lambda record: record.created_at, reverse=True)
        if limit is not None:
            return records[:limit]
        return records

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        category: MemoryCategory | str | None = None,
        tier: MemoryTier | str | None = None,
        limit: int | None = None,
        include_expired: bool = False,
        include_resolved: bool = False,
        include_history: bool = False,
        now: datetime | None = None,
    ) -> list[MemoryRecord]:
        """Return ranked whole-word matches after applying all filters.

        Expired and resolved tasks are omitted by default without deleting history.
        """
        if limit is not None and limit < 0:
            raise ValueError("Search limit must be nonnegative")
        if not tokenize(query) or limit == 0:
            return []

        candidates = self._filter(
            self._visible(include_expired=include_expired, include_resolved=include_resolved, include_history=include_history, now=now),
            user_id=user_id,
            session_id=session_id,
            category=category,
            tier=tier,
        )
        records = self.ranker.rank(query, candidates, now=now)
        if limit is not None:
            return records[:limit]
        return records

    @staticmethod
    def _filter(
        records: Iterable[MemoryRecord],
        *,
        user_id: str | None,
        session_id: str | None,
        category: MemoryCategory | str | None,
        tier: MemoryTier | str | None,
    ) -> list[MemoryRecord]:
        category_value = category.value if isinstance(category, MemoryCategory) else category
        tier_value = tier.value if isinstance(tier, MemoryTier) else tier

        filtered: list[MemoryRecord] = []
        for record in records:
            if user_id is not None and record.user_id != user_id:
                continue
            if session_id is not None and record.session_id != session_id and session_id not in record.evidence_session_ids:
                continue
            if category_value is not None and record.category.value != category_value:
                continue
            if tier_value is not None and record.tier.value != tier_value:
                continue
            filtered.append(record)
        return filtered
