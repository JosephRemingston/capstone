"""Future storage/retrieval integration contracts."""

from __future__ import annotations

from typing import Protocol

from .models import MemoryRecord


class GraphMemoryAdapter(Protocol):
    """Placeholder contract for a later graph layer.

    This phase intentionally does not implement graph schemas, Neo4j clients,
    graph traversal, or graph persistence. Future adapters can implement this
    protocol without changing the Memory Core API.
    """

    def index(self, record: MemoryRecord) -> None:
        """Index a normalized record in a graph-backed memory store."""
