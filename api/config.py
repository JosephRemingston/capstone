"""Configuration for the CogniMem HTTP application."""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    store_path: Path
    auth_db_path: Path
    index_path: Path
    jwt_secret: str
    access_token_minutes: int = 30
    cors_origins: tuple[str, ...] = ()
    log_level: str = "INFO"
    ann_enabled: bool = True
    ann_exact_threshold: int = 256

    @classmethod
    def from_env(cls) -> "Settings":
        store = Path(os.getenv("COGNIMEM_STORE", "memory_store/memories.jsonl"))
        auth = Path(os.getenv("COGNIMEM_AUTH_DB", "memory_store/auth.sqlite3"))
        index = Path(os.getenv("COGNIMEM_INDEX", str(store.with_suffix(".index.sqlite3"))))
        secret = os.getenv("COGNIMEM_JWT_SECRET", "")
        if len(secret.encode("utf-8")) < 32:
            raise ValueError("COGNIMEM_JWT_SECRET must contain at least 32 bytes")
        minutes = int(os.getenv("COGNIMEM_ACCESS_TOKEN_MINUTES", "30"))
        if not 5 <= minutes <= 1440:
            raise ValueError("COGNIMEM_ACCESS_TOKEN_MINUTES must be between 5 and 1440")
        origins = tuple(item.strip() for item in os.getenv("COGNIMEM_CORS_ORIGINS", "").split(",") if item.strip())
        threshold = int(os.getenv("COGNIMEM_ANN_EXACT_THRESHOLD", "256"))
        if threshold < 0:
            raise ValueError("COGNIMEM_ANN_EXACT_THRESHOLD must be nonnegative")
        return cls(
            store_path=store,
            auth_db_path=auth,
            index_path=index,
            jwt_secret=secret,
            access_token_minutes=minutes,
            cors_origins=origins,
            log_level=os.getenv("COGNIMEM_LOG_LEVEL", "INFO").upper(),
            ann_enabled=os.getenv("COGNIMEM_ANN_ENABLED", "true").lower() in {"1", "true", "yes", "on"},
            ann_exact_threshold=threshold,
        )

    @classmethod
    def for_testing(cls, root: Path, *, jwt_secret: str | None = None) -> "Settings":
        return cls(
            store_path=root / "memories.jsonl",
            auth_db_path=root / "auth.sqlite3",
            index_path=root / "index.sqlite3",
            jwt_secret=jwt_secret or secrets.token_urlsafe(48),
            cors_origins=(),
            ann_enabled=True,
            ann_exact_threshold=8,
        )
