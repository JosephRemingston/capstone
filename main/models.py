"""Core memory data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class MemoryCategory(str, Enum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    PREFERENCE = "preference"
    TASK = "task"
    TEMPORARY = "temporary"


class MemoryTier(str, Enum):
    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    ARCHIVE = "archive"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _serialize_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(slots=True)
class MemoryInput:
    """Raw incoming interaction and caller-provided metadata."""

    content: str
    user_id: str
    session_id: str
    role: str = "user"
    timestamp: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content or not self.content.strip():
            raise ValueError("MemoryInput.content must not be empty")
        if not self.user_id:
            raise ValueError("MemoryInput.user_id must not be empty")
        if not self.session_id:
            raise ValueError("MemoryInput.session_id must not be empty")
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)


@dataclass(slots=True)
class MemoryRecord:
    """Normalized memory object created by the Memory Core."""

    content: str
    user_id: str
    session_id: str
    category: MemoryCategory
    tier: MemoryTier = MemoryTier.WORKING
    id: str = field(default_factory=lambda: str(uuid4()))
    role: str = "user"
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    confidence: float = 1.0
    importance_score: float = 0.0
    access_count: int = 0
    source_metadata: dict[str, Any] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    expires_at: datetime | None = None
    archive_after: datetime | None = None
    due_at: datetime | None = None
    task_status: str | None = None
    related_task_id: str | None = None
    last_accessed_at: datetime | None = None

    @classmethod
    def from_input(
        cls,
        memory_input: MemoryInput,
        category: MemoryCategory,
        confidence: float = 1.0,
    ) -> "MemoryRecord":
        return cls(
            content=memory_input.content.strip(),
            user_id=memory_input.user_id,
            session_id=memory_input.session_id,
            category=category,
            role=memory_input.role,
            created_at=memory_input.timestamp,
            updated_at=memory_input.timestamp,
            confidence=confidence,
            source_metadata=dict(memory_input.metadata),
        )

    def touch(self, when: datetime | None = None) -> None:
        self.access_count += 1
        self.updated_at = when or utc_now()
        if self.updated_at.tzinfo is None:
            self.updated_at = self.updated_at.replace(tzinfo=timezone.utc)
        self.last_accessed_at = self.updated_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_count": self.access_count,
            "archive_after": _serialize_datetime(self.archive_after),
            "category": self.category.value,
            "confidence": self.confidence,
            "content": self.content,
            "created_at": _serialize_datetime(self.created_at),
            "expires_at": _serialize_datetime(self.expires_at),
            "features": dict(self.features),
            "id": self.id,
            "importance_score": self.importance_score,
            "role": self.role,
            "session_id": self.session_id,
            "source_metadata": dict(self.source_metadata),
            "tier": self.tier.value,
            "updated_at": _serialize_datetime(self.updated_at),
            "user_id": self.user_id,
            "due_at": _serialize_datetime(self.due_at),
            "task_status": self.task_status,
            "related_task_id": self.related_task_id,
            "last_accessed_at": _serialize_datetime(self.last_accessed_at),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryRecord":
        return cls(
            access_count=int(payload.get("access_count", 0)),
            archive_after=_parse_datetime(payload.get("archive_after")),
            category=MemoryCategory(payload["category"]),
            confidence=float(payload.get("confidence", 1.0)),
            content=payload["content"],
            created_at=_parse_datetime(payload.get("created_at")) or utc_now(),
            expires_at=_parse_datetime(payload.get("expires_at")),
            features=dict(payload.get("features", {})),
            id=payload["id"],
            importance_score=float(payload.get("importance_score", 0.0)),
            role=payload.get("role", "user"),
            session_id=payload["session_id"],
            source_metadata=dict(payload.get("source_metadata", {})),
            tier=MemoryTier(payload.get("tier", MemoryTier.WORKING.value)),
            updated_at=_parse_datetime(payload.get("updated_at")) or utc_now(),
            user_id=payload["user_id"],
            due_at=_parse_datetime(payload.get("due_at")),
            task_status=payload.get("task_status"),
            related_task_id=payload.get("related_task_id"),
            last_accessed_at=_parse_datetime(payload.get("last_accessed_at")),
        )
