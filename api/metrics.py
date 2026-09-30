"""Prometheus metrics used by the HTTP layer."""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class Metrics:
    def __init__(self, registry: CollectorRegistry) -> None:
        self.registry = registry
        self.requests = Counter(
            "cognimem_http_requests_total", "HTTP requests received", ["method", "route", "status"], registry=registry
        )
        self.latency = Histogram(
            "cognimem_http_request_duration_seconds", "HTTP request latency", ["method", "route"], registry=registry
        )
        self.memories = Counter(
            "cognimem_memories_ingested_total", "Memory observations ingested", registry=registry
        )
        self.auth = Counter(
            "cognimem_auth_events_total", "Authentication events", ["event", "status"], registry=registry
        )
        self.ready = Gauge("cognimem_ready", "Readiness status", registry=registry)
