"""Local JSONL-backed memory store."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .models import MemoryCategory, MemoryRecord, MemoryTier
from .retrieval import MemoryRanker, tokenize


DEFAULT_STORE_PATH = Path("memory_store") / "memories.jsonl"


@dataclass(slots=True)
class LocalMemoryStore:
    """Append-only local storage for serialized memory records."""

    path: Path | str = DEFAULT_STORE_PATH
    ranker: MemoryRanker = field(default_factory=MemoryRanker)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def save(self, record: MemoryRecord) -> MemoryRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")
        return record

    def all(self) -> list[MemoryRecord]:
        if not self.path.exists():
            return []

        records: list[MemoryRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                    records.append(MemoryRecord.from_dict(payload))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid memory record on line {line_number} in {self.path}") from exc
        return records

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
    ) -> list[MemoryRecord]:
        records = self._filter(
            self.all(),
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
    ) -> list[MemoryRecord]:
        """Return ranked whole-word matches after applying all filters.

        Expiry timestamps remain hints; search does not expire or mutate records.
        """
        if limit is not None and limit < 0:
            raise ValueError("Search limit must be nonnegative")
        if not tokenize(query) or limit == 0:
            return []

        candidates = self._filter(
            self.all(),
            user_id=user_id,
            session_id=session_id,
            category=category,
            tier=tier,
        )
        records = self.ranker.rank(query, candidates)
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
            if session_id is not None and record.session_id != session_id:
                continue
            if category_value is not None and record.category.value != category_value:
                continue
            if tier_value is not None and record.tier.value != tier_value:
                continue
            filtered.append(record)
        return filtered
