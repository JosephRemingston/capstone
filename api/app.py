"""FastAPI application for the CogniMem production-facing HTTP surface."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

from .auth import AuthConfig, AuthManager, AuthStore, current_user, require_roles
from .config import Settings
from .logging import configure_logging
from .metrics import Metrics
from .schemas import AuthResponse, LoginRequest, MemoryCreateRequest, SearchRequest, SignupRequest
from .services import MemoryService
from main.domain.models import MemoryCategory, MemoryTier
from main.storage.store import LocalMemoryStore


logger = logging.getLogger("cognimem.api")


def success(data, request_id: str, status_code: int = 200) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"success": True, "data": data, "error": None, "request_id": request_id})


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)
    app = FastAPI(title="CogniMem API", version="1.0.0", docs_url="/docs", redoc_url="/redoc")
    app.state.settings = settings
    app.state.auth_store = AuthStore(settings.auth_db_path)
    app.state.auth_manager = AuthManager(
        app.state.auth_store, AuthConfig(settings.jwt_secret, settings.access_token_minutes)
    )
    app.state.store = LocalMemoryStore(settings.store_path)
    app.state.service = MemoryService.create(
        app.state.store,
        index_path=settings.index_path,
        ann_enabled=settings.ann_enabled,
        ann_exact_threshold=settings.ann_exact_threshold,
    )
    app.state.registry = CollectorRegistry()
    app.state.metrics = Metrics(app.state.registry)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @app.middleware("http")
    async def request_middleware(request: Request, call_next):
        supplied_request_id = request.headers.get("X-Request-ID", "").strip()
        request_id = supplied_request_id if len(supplied_request_id) <= 128 and supplied_request_id.replace("-", "").replace("_", "").replace(":", "").replace(".", "").isalnum() else str(uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        status_code = 500
        route = request.url.path
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            duration = time.perf_counter() - started
            app.state.metrics.requests.labels(request.method, route, str(status_code)).inc()
            app.state.metrics.latency.labels(request.method, route).observe(duration)
            logger.info(
                "http_request",
                extra={"request_id": request_id, "method": request.method, "path": route,
                       "status_code": status_code, "duration_ms": round(duration * 1000, 3)},
            )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={
            "success": False, "data": None,
            "error": {"code": "invalid_input", "message": str(exc)},
            "request_id": getattr(request.state, "request_id", str(uuid4())),
        })

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={
            "success": False, "data": None,
            "error": {"code": "validation_error", "message": "Request validation failed", "details": exc.errors()},
            "request_id": getattr(request.state, "request_id", str(uuid4())),
        })

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={
            "success": False, "data": None,
            "error": {"code": "http_error", "message": exc.detail},
            "request_id": getattr(request.state, "request_id", str(uuid4())),
        }, headers=exc.headers)

    @app.get("/health/live")
    def live(request: Request):
        return success({"status": "ok"}, request.state.request_id)

    @app.get("/health/ready")
    def ready(request: Request):
        checks = {}
        try:
            request.app.state.auth_store.health()
            checks["auth_store"] = "ok"
        except Exception:
            checks["auth_store"] = "error"
        try:
            request.app.state.store._read_log()
            checks["memory_store"] = "ok"
        except Exception:
            checks["memory_store"] = "error"
        try:
            with request.app.state.service.rag.database.connect() as db:
                db.execute("SELECT 1").fetchone()
            checks["index_store"] = "ok"
        except Exception:
            checks["index_store"] = "error"
        ready_status = all(value == "ok" for value in checks.values())
        request.app.state.metrics.ready.set(1 if ready_status else 0)
        body = {"status": "ready" if ready_status else "not_ready", "checks": checks}
        return success(body, request.state.request_id, 200 if ready_status else 503)

    @app.get("/metrics")
    def metrics(request: Request):
        payload = generate_latest(request.app.state.registry)
        return Response(content=payload, media_type=CONTENT_TYPE_LATEST)

    @app.post("/api/v1/auth/signup", response_model=dict, status_code=201)
    def signup(request: Request, body: SignupRequest):
        try:
            user = request.app.state.auth_store.create_user(body.email, body.password)
        except ValueError as exc:
            request.app.state.metrics.auth.labels("signup", "rejected").inc()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        token = request.app.state.auth_manager.token_for(user)
        request.app.state.metrics.auth.labels("signup", "accepted").inc()
        data = AuthResponse(access_token=token, expires_in=request.app.state.settings.access_token_minutes * 60, user=user).model_dump()
        return success(data, request.state.request_id, 201)

    @app.post("/api/v1/auth/login", response_model=dict)
    def login(request: Request, body: LoginRequest):
        try:
            user = request.app.state.auth_store.authenticate(body.email, body.password)
        except ValueError:
            user = None
        if not user:
            request.app.state.metrics.auth.labels("login", "rejected").inc()
            raise HTTPException(status_code=401, detail="Invalid email or password")
        token = request.app.state.auth_manager.token_for(user)
        request.app.state.metrics.auth.labels("login", "accepted").inc()
        data = AuthResponse(access_token=token, expires_in=request.app.state.settings.access_token_minutes * 60,
                            user={"id": user["id"], "email": user["email"], "role": user["role"], "created_at": user["created_at"]}).model_dump()
        return success(data, request.state.request_id)

    @app.get("/api/v1/me")
    def me(request: Request, user: dict = Depends(current_user)):
        return success({"id": user["id"], "email": user["email"], "role": user["role"], "created_at": user["created_at"]}, request.state.request_id)

    @app.post("/api/v1/memories", status_code=201)
    def create_memory(request: Request, body: MemoryCreateRequest, user: dict = Depends(require_roles("user", "admin"))):
        record = request.app.state.service.ingest(user_id=user["id"], payload=body.model_dump())
        request.app.state.metrics.memories.inc()
        return success(record.to_dict(), request.state.request_id, 201)

    @app.get("/api/v1/memories")
    def list_memories(request: Request, limit: int = 20, session_id: str | None = None,
                      category: str | None = None, tier: str | None = None,
                      include_history: bool = False, user: dict = Depends(require_roles("user", "admin"))):
        if not 1 <= limit <= 100:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
        if category is not None:
            MemoryCategory(category)
        if tier is not None:
            MemoryTier(tier)
        records = request.app.state.store.list(user_id=user["id"], session_id=session_id, category=category,
                                               tier=tier, limit=limit, include_history=include_history)
        return success([record.to_dict() for record in records], request.state.request_id)

    @app.get("/api/v1/memories/{record_id}")
    def get_memory(record_id: str, request: Request, user: dict = Depends(require_roles("user", "admin"))):
        record = request.app.state.store.get(record_id)
        if record is None or record.user_id != user["id"]:
            raise HTTPException(status_code=404, detail="Memory not found")
        return success(record.to_dict(), request.state.request_id)

    @app.post("/api/v1/memories/search")
    def search_memories(request: Request, body: SearchRequest, user: dict = Depends(require_roles("user", "admin"))):
        if body.category is not None:
            MemoryCategory(body.category)
        if body.tier is not None:
            MemoryTier(body.tier)
        if body.mode == "keyword":
            records = request.app.state.store.search(body.query, user_id=user["id"], session_id=body.session_id,
                                                     category=body.category, tier=body.tier, limit=body.limit)
            data = [record.to_dict() for record in records]
        else:
            data = request.app.state.service.rag.search(
                body.query, user_id=user["id"], mode=body.mode, limit=body.limit,
                candidate_limit=body.candidate_limit, rerank=body.rerank,
                session_id=body.session_id, category=body.category, tier=body.tier,
            )
        return success(data, request.state.request_id)

    @app.get("/api/v1/me/export")
    def export_data(request: Request, format: str = "json", user: dict = Depends(require_roles("user", "admin"))):
        if format not in {"json", "csv"}:
            raise HTTPException(status_code=422, detail="format must be json or csv")
        body, media_type, filename = request.app.state.service.export_user(user, format)
        return StreamingResponse(iter([body]), media_type=media_type,
                                 headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    @app.delete("/api/v1/me")
    def delete_account(request: Request, user: dict = Depends(require_roles("user", "admin"))):
        result = request.app.state.service.delete_user_data(user["id"])
        if not request.app.state.auth_store.delete_user(user["id"]):
            raise HTTPException(status_code=500, detail="Account deletion could not be completed")
        return success(result, request.state.request_id)

    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.exists():
        app.mount("/demo", StaticFiles(directory=static_dir, html=True), name="demo")

    return app


try:
    app = create_app()
except ValueError:
    # Import remains safe for CLI/tests that do not configure the HTTP service.
    app = None
