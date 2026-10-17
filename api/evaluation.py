"""Read-only access to the tracked evaluation snapshots (evaluation/reports/<tag>_summary.json) for the web app.

Every number comes from files written by scripts/measure_all.py and training/ensemble; nothing is recomputed here.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TAG = re.compile(r"[A-Za-z0-9_-]{1,64}")
SUFFIX = "_summary.json"
MANIFEST_FIELDS = ("tag", "created_at", "git_head", "dirty_tree", "dataset_sha256", "scorer", "steps", "reused")
MODEL_FIELDS = ("generated_at", "model", "dataset", "class_names", "resamples", "class_values", "tuned",
                "ensemble_thresholds", "majority", "oracle", "serving_checks", "caveats")


def reports_dir() -> Path:
    return Path(os.getenv("COGNIMEM_REPORTS_DIR") or ROOT / "evaluation" / "reports")


def _read(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)   # bool is an int subclass


def _snapshot_files() -> dict[str, Path]:
    found = {}
    for path in sorted(reports_dir().glob(f"*{SUFFIX}")):
        tag = path.name[: -len(SUFFIX)]
        if TAG.fullmatch(tag):
            found[tag] = path
    return found


def list_snapshots() -> list[dict[str, Any]]:
    """Every snapshot summary that has metrics (retention_summary.json is not a snapshot), oldest first."""
    rows = []
    for tag, path in _snapshot_files().items():
        data = _read(path)
        if not data or not isinstance(data.get("flat"), dict):
            continue
        manifest = data.get("manifest") or {}
        rows.append({"tag": tag, "created_at": manifest.get("created_at"), "git_head": manifest.get("git_head"),
                     "scorer": manifest.get("scorer"), "sections": [s for s in data.get("sections", {}) if not s.startswith("_")],
                     "metrics": sum(_is_number(v) for v in data["flat"].values())})
    return sorted(rows, key=lambda row: (row["created_at"] or "", row["tag"]))


def load_snapshot(tag: str) -> dict[str, Any] | None:
    """One snapshot's numeric metrics (flat keys, as scripts/report_delta.py compares them) and its provenance."""
    if not TAG.fullmatch(tag):
        return None
    path = _snapshot_files().get(tag)
    data = _read(path) if path else None
    if not data or not isinstance(data.get("flat"), dict):
        return None
    manifest = data.get("manifest") or {}
    flat = {key: value for key, value in data["flat"].items() if value is None or _is_number(value)}
    return {"tag": tag, "manifest": {key: manifest.get(key) for key in MANIFEST_FIELDS}, "flat": flat}


def load_model_report() -> dict[str, Any] | None:
    """The default importance model's training report (training/ensemble), without per-seed and file details."""
    data = _read(reports_dir() / "retention_ensemble.json")
    return {key: data.get(key) for key in MODEL_FIELDS} if data else None
