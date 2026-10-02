"""Application services bridging HTTP contracts to the existing memory core."""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from main.application.core import MemoryCore
from main.domain.models import MemoryInput, MemoryRecord
from main.retrieval.rag import MemoryRAG
from main.storage.cleanup import atomic_write, purge_indexes
from main.storage.store import LocalMemoryStore
from main.storage.locking import store_lock


@dataclass(slots=True)
class MemoryService:
    store: LocalMemoryStore
    core: MemoryCore
    rag: MemoryRAG

    @classmethod
    def create(cls, store: LocalMemoryStore, *, index_path: Path, ann_enabled: bool = True, ann_exact_threshold: int = 256):
        rag = MemoryRAG(store, index_path=index_path, ann_enabled=ann_enabled, ann_exact_threshold=ann_exact_threshold)
        return cls(store=store, core=MemoryCore(), rag=rag)

    def ingest(self, *, user_id: str, payload: dict[str, Any]) -> MemoryRecord:
        metadata = dict(payload.get("metadata") or {})
        timestamp = payload.get("timestamp") or datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        memory_input = MemoryInput(
            content=payload["content"],
            user_id=user_id,
            session_id=payload["session_id"],
            role=payload.get("role", "user"),
            timestamp=timestamp,
            metadata=metadata,
        )
        record = self.core.process(memory_input)
        saved = self.store.ingest(record)
        return saved

    def export_user(self, user: dict[str, Any], fmt: str) -> tuple[bytes, str, str]:
        user_id = user["id"]
        records = self.store._read_log()
        owned = [record.to_dict() for record in records if record.user_id == user_id]
        account = {"id": user_id, "email": user["email"], "role": user["role"], "created_at": user["created_at"]}
        if fmt == "json":
            body = json.dumps({"account": account, "memories": owned}, ensure_ascii=False, indent=2).encode("utf-8")
            return body, "application/json", f"cognimem-{user_id}.json"
        if fmt != "csv":
            raise ValueError("format must be json or csv")
        fields = ["id", "content", "session_id", "role", "category", "tier", "created_at", "updated_at",
                  "confidence", "importance_score", "memory_status", "task_status"]
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in owned:
            row = dict(record)
            row["account_email"] = account["email"]
            row["account_role"] = account["role"]
            writer.writerow(row)
        return output.getvalue().encode("utf-8"), "text/csv; charset=utf-8", f"cognimem-{user_id}.csv"

    def delete_user_data(self, user_id: str) -> dict[str, int]:
        """Atomically rewrite the authoritative log without this user's observations, then purge indexes."""
        with store_lock(self.store.path):
            revisions = self.store._read_log()
            remaining = [record for record in revisions if record.user_id != user_id]
            removed = len(revisions) - len(remaining)
            payload = "".join(
                json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n" for record in remaining
            ).encode("utf-8")
            # Purge derived indexes regardless of whether the source log currently
            # contains rows. This also removes stale derived data left by an earlier
            # interrupted operation.
            purge_indexes(self.store, user_id)
            if removed:
                atomic_write(self.store.path, payload)
            return {"memories_deleted": removed}
