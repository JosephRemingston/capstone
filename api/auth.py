"""Authentication, password hashing, JWT handling, and role checks."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
import sqlite3
from pathlib import Path
from typing import Callable, Iterable
from uuid import uuid4

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHash, VerificationError, VerifyMismatchError
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PASSWORD_MIN_LENGTH = 12


class AuthStore:
    """Small SQLite identity store; memory observations remain in the existing JSONL store."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user','admin')) DEFAULT 'user',
                    created_at TEXT NOT NULL,
                    disabled INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS users_email ON users(email);
                """
            )

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    @staticmethod
    def normalize_email(email: str) -> str:
        email = email.strip().casefold()
        if not EMAIL_RE.fullmatch(email) or len(email) > 320:
            raise ValueError("A valid email address is required")
        return email

    @staticmethod
    def validate_password(password: str) -> None:
        if len(password) < PASSWORD_MIN_LENGTH or len(password) > 256:
            raise ValueError("Password must be between 12 and 256 characters")

    def create_user(self, email: str, password: str, role: str = "user") -> dict:
        email = self.normalize_email(email)
        self.validate_password(password)
        if role not in {"user", "admin"}:
            raise ValueError("Invalid role")
        user_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        password_hash = PasswordHasher().hash(password)
        try:
            with self.connect() as db:
                db.execute(
                    "INSERT INTO users(id,email,password_hash,role,created_at) VALUES(?,?,?,?,?)",
                    (user_id, email, password_hash, role, now),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("An account with that email already exists") from exc
        return {"id": user_id, "email": email, "role": role, "created_at": now}

    def authenticate(self, email: str, password: str) -> dict | None:
        email = self.normalize_email(email)
        try:
            with self.connect() as db:
                row = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
            if row is None or row["disabled"]:
                return None
            PasswordHasher().verify(row["password_hash"], password)
        except (VerifyMismatchError, VerificationError, InvalidHash, ValueError):
            return None
        return dict(row)

    def get_user(self, user_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT id,email,role,created_at,disabled FROM users WHERE id=?", (user_id,)).fetchone()
        return dict(row) if row else None

    def delete_user(self, user_id: str) -> bool:
        with self.connect() as db:
            result = db.execute("DELETE FROM users WHERE id=?", (user_id,))
            return result.rowcount == 1

    def health(self) -> None:
        with self.connect() as db:
            db.execute("SELECT 1").fetchone()


@dataclass(frozen=True, slots=True)
class AuthConfig:
    secret: str
    access_token_minutes: int = 30
    algorithm: str = "HS256"

    def __post_init__(self) -> None:
        if len(self.secret.encode("utf-8")) < 32:
            raise ValueError("JWT secret must contain at least 32 bytes")
        if not 5 <= self.access_token_minutes <= 1440:
            raise ValueError("Access token lifetime must be between 5 and 1440 minutes")


class AuthManager:
    def __init__(self, store: AuthStore, config: AuthConfig) -> None:
        self.store = store
        self.config = config
        self.password_hasher = PasswordHasher()

    def token_for(self, user: dict) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user["id"],
            "role": user["role"],
            "iat": now,
            "exp": now + timedelta(minutes=self.config.access_token_minutes),
            "typ": "access",
        }
        return jwt.encode(payload, self.config.secret, algorithm=self.config.algorithm)

    def decode(self, token: str) -> dict:
        try:
            payload = jwt.decode(
                token,
                self.config.secret,
                algorithms=[self.config.algorithm],
                options={"require": ["sub", "role", "iat", "exp", "typ"]},
            )
        except jwt.PyJWTError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired access token") from exc
        if payload.get("typ") != "access" or payload.get("role") not in {"user", "admin"}:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid access token")
        user = self.store.get_user(str(payload["sub"]))
        if not user or user["disabled"] or user["role"] != payload["role"]:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account is unavailable")
        return user


bearer = HTTPBearer(auto_error=False)


def current_user(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
    manager: AuthManager = request.app.state.auth_manager
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Bearer authentication is required")
    return manager.decode(credentials.credentials)


def require_roles(*roles: str) -> Callable:
    allowed = set(roles)

    def dependency(user: dict = Depends(current_user)) -> dict:
        if user["role"] not in allowed:
            raise HTTPException(status_code=403, detail="Insufficient role")
        return user

    return dependency
