"""Validated HTTP request/response contracts."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Envelope(BaseModel):
    success: bool = True
    data: Any = None
    error: dict[str, Any] | None = None
    request_id: str


class SignupRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=12, max_length=256)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=256)


class AuthResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    user: dict[str, Any]


class MemoryCreateRequest(BaseModel):
    content: str = Field(min_length=1, max_length=100_000)
    session_id: str = Field(min_length=1, max_length=256)
    role: str = Field(default="user", min_length=1, max_length=32)
    timestamp: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content", "session_id", "role")
    @classmethod
    def strip_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    mode: Literal["keyword", "semantic", "hybrid", "graph"] = "hybrid"
    limit: int = Field(default=5, ge=1, le=20)
    budget: int = Field(default=10_000, ge=256, le=100_000)
    candidate_limit: int | None = Field(default=None, ge=1, le=200)
    rerank: bool | None = None
    # Extractive by default: memory text leaves the server only when the caller asks for Gemini.
    generator: Literal["extractive", "gemini"] = "extractive"
    abstain: bool = False
    as_of: str | None = Field(default=None, max_length=64)
    known_at: str | None = Field(default=None, max_length=64)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value

    @field_validator("as_of", "known_at")
    @classmethod
    def iso_time(cls, value: str | None) -> str | None:
        if value is not None:
            from main.graph import parse_time
            parse_time(value)   # ValueError -> 422 here, never mistaken for a provider outage
        return value

    @model_validator(mode="after")
    def abstain_needs_extractive(self) -> "AskRequest":
        if self.abstain and self.generator != "extractive":
            raise ValueError("abstain works only with the extractive generator")
        return self


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    mode: Literal["keyword", "semantic", "hybrid", "graph", "recency"] = "hybrid"
    limit: int = Field(default=10, ge=1, le=100)
    candidate_limit: int | None = Field(default=None, ge=1, le=200)
    rerank: bool | None = None
    session_id: str | None = Field(default=None, max_length=256)
    category: str | None = None
    tier: str | None = None

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class ExportQuery(BaseModel):
    format: Literal["json", "csv"] = "json"


class HealthResponse(BaseModel):
    status: Literal["ok", "ready", "not_ready"]
    checks: dict[str, str]
