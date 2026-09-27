"""Storage-independent temporal graph integration contract."""
from __future__ import annotations
from datetime import datetime
from typing import Protocol
from .models import MemoryRecord


class GraphMemoryAdapter(Protocol):
    """Implemented by TemporalGraph; callers pass the authoritative revision log."""
    def sync(self, revisions: list[MemoryRecord], *, user_id: str) -> dict: ...
    def entities(self, query: str, *, user_id: str, known_at: datetime | None = None, exact=False) -> list[dict]: ...
    def relations(self, *, user_id: str, as_of: datetime | None = None, known_at: datetime | None = None,
                  entity_id: str | None = None, predicate: str | None = None, include_negative=True) -> list[dict]: ...
    def traverse(self, seeds: list[str], *, user_id: str, as_of=None, known_at=None, max_hops=2,
                 max_paths=100, allowed_ids: set[str] | None = None, direction='both') -> list[dict]: ...
