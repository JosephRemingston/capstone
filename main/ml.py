"""Optional XGBoost importance scoring for the Hippocorpus proxy target."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .features import FeatureExtractor
from .models import MemoryInput, MemoryRecord

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "importance"
FEATURE_VERSION = "hippocorpus-text-v1"
TEXT_FEATURES = (
    "category_episodic", "category_preference", "category_procedural",
    "category_semantic", "category_task", "category_temporary", "entity_density",
    "has_deadline", "preference_signal", "sentiment_strength", "task_signal",
    "word_count_norm", "text_length_log",
)
FEATURE_NAMES = list(TEXT_FEATURES) + [f"lexical_{i}" for i in range(512)]


@dataclass
class MLFeatureExtractor(FeatureExtractor):
    """Add deterministic hashed word counts, with no fitted vocabulary or labels."""

    legacy: bool = True

    def extract(self, memory_input: MemoryInput, record: MemoryRecord) -> dict[str, float]:
        features = super().extract(memory_input, record)
        words = re.findall(r"\b[\w'-]+\b", memory_input.content.casefold())
        counts = Counter(int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big") % 512 for word in words)
        features["text_length_log"] = math.log1p(len(words))
        features.update({f"lexical_{i}": math.log1p(counts.get(i, 0)) for i in range(512)})
        return features


class MLImportanceScorer:
    """Load a native XGBoost artifact and implement the existing score interface.

    This predicts personal-event significance, not validated conversational
    retention utility. Dependencies are imported only when this scorer is used.
    """

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR) -> None:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ValueError("Install requirements-ml.txt to use the ML scorer") from exc
        directory = Path(model_dir)
        try:
            self.metadata = json.loads((directory / "metadata.json").read_text())
            if self.metadata["feature_version"] != FEATURE_VERSION or self.metadata["feature_names"] != FEATURE_NAMES:
                raise ValueError("Incompatible ML feature schema")
            model_bytes = (directory / "model.ubj").read_bytes()
            if hashlib.sha256(model_bytes).hexdigest() != self.metadata["model_sha256"]:
                raise ValueError("ML model checksum mismatch")
            self.model = xgb.Booster()
            self.model.load_model(bytearray(model_bytes))
            self.model.set_param({"nthread": 1})
        except (OSError, KeyError, json.JSONDecodeError, xgb.core.XGBoostError) as exc:
            raise ValueError(f"Cannot load importance model from {directory}: {exc}") from exc

    def score(self, features: dict[str, float]) -> float:
        import numpy as np
        missing = set(FEATURE_NAMES) - features.keys()
        if missing:
            raise ValueError("ML scorer requires MLFeatureExtractor features")
        values = [float(features[name]) for name in FEATURE_NAMES]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("ML features must be finite")
        prediction = float(self.model.inplace_predict(np.asarray([values], dtype=np.float32))[0])
        if not math.isfinite(prediction):
            raise ValueError("ML model returned a nonfinite prediction")
        return round(max(0.0, min(prediction, 1.0)), 4)
