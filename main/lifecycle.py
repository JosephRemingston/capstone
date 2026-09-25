"""Memory lifecycle tier assignment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .models import MemoryCategory, MemoryRecord, MemoryTier
from .text_rules import recurring


@dataclass(slots=True)
class LifecycleManager:
    """Assigns memories to working, short-term, long-term, or archive tiers."""

    working_threshold: float = 0.25
    long_term_threshold: float = 0.42
    archive_age_days: int = 90

    def assign_tier(self, record: MemoryRecord, score: float) -> MemoryTier:
        if self._should_archive(record, score):
            return MemoryTier.ARCHIVE

        if record.category is MemoryCategory.TEMPORARY or score < self.working_threshold:
            return MemoryTier.WORKING

        # A one-off action can matter greatly without being a durable user fact.
        if record.category is MemoryCategory.TASK and not recurring(record.content):
            return MemoryTier.SHORT_TERM

        durable_categories = {
            MemoryCategory.PREFERENCE,
            MemoryCategory.PROCEDURAL,
            MemoryCategory.SEMANTIC,
            MemoryCategory.TASK,
        }
        if record.category in durable_categories and score >= self.long_term_threshold:
            return MemoryTier.LONG_TERM

        return MemoryTier.SHORT_TERM

    def expiry_for(self, record: MemoryRecord) -> datetime | None:
        if record.tier is MemoryTier.WORKING:
            return record.created_at + timedelta(hours=1)
        if record.tier is MemoryTier.SHORT_TERM:
            return record.created_at + timedelta(days=14)
        return None

    def archive_after_for(self, record: MemoryRecord) -> datetime | None:
        if record.tier is MemoryTier.LONG_TERM:
            return record.created_at + timedelta(days=self.archive_age_days)
        return None

    def _should_archive(self, record: MemoryRecord, score: float) -> bool:
        if record.tier is MemoryTier.ARCHIVE:
            return True

        age = record.updated_at - record.created_at
        stale = age >= timedelta(days=self.archive_age_days)
        historically_useful = record.access_count > 0 or score >= self.long_term_threshold
        return stale and historically_useful
