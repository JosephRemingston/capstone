"""Optional local neural retention-proxy scorer."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .ml import TEXT_FEATURES


DEFAULT_NEURAL_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "importance_neural"
SCORE_VALUES = np.asarray((0.0, 0.5, 1.0), dtype=np.float32)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ValueError("Install requirements-neural.txt to use the neural scorer") from exc
    return torch


def create_head(architecture: str = "frozen-bge-deep-head-v1"):
    """Return a trainable head for the requested frozen-encoder architecture."""
    torch = _torch()
    if architecture not in {"frozen-bge-two-branch-v1", "frozen-bge-deep-head-v1"}:
        raise ValueError("Unknown neural architecture")

    class RetentionHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if architecture == "frozen-bge-deep-head-v1":
                self.text_branch = torch.nn.Sequential(
                    torch.nn.Linear(384, 192), torch.nn.LayerNorm(192),
                    torch.nn.GELU(), torch.nn.Dropout(0.2),
                    torch.nn.Linear(192, 128), torch.nn.LayerNorm(128),
                    torch.nn.GELU(), torch.nn.Dropout(0.2),
                )
                self.numeric_branch = torch.nn.Sequential(
                    torch.nn.Linear(len(TEXT_FEATURES), 64), torch.nn.LayerNorm(64),
                    torch.nn.GELU(), torch.nn.Linear(64, 32),
                    torch.nn.LayerNorm(32), torch.nn.GELU(),
                )
                self.head = torch.nn.Sequential(
                    torch.nn.Linear(160, 128), torch.nn.GELU(),
                    torch.nn.Dropout(0.2), torch.nn.Linear(128, 64),
                    torch.nn.GELU(), torch.nn.Dropout(0.2),
                    torch.nn.Linear(64, 3),
                )
            else:
                self.text_branch = torch.nn.Sequential(
                    torch.nn.Linear(384, 128), torch.nn.LayerNorm(128),
                    torch.nn.GELU(), torch.nn.Dropout(0.2),
                )
                self.numeric_branch = torch.nn.Sequential(
                    torch.nn.Linear(len(TEXT_FEATURES), 32), torch.nn.LayerNorm(32),
                    torch.nn.GELU(),
                )
                self.head = torch.nn.Sequential(
                    torch.nn.Linear(160, 64), torch.nn.GELU(),
                    torch.nn.Dropout(0.2), torch.nn.Linear(64, 3),
                )

        def forward(self, embedding, numeric):
            if embedding.ndim != 2 or embedding.shape[1] != 384:
                raise ValueError("Expected 384-dimensional text embeddings")
            if numeric.ndim != 2 or numeric.shape[1] != len(TEXT_FEATURES):
                raise ValueError("Expected 13 numeric features")
            combined = torch.cat((self.text_branch(embedding),
                                  self.numeric_branch(numeric)), dim=1)
            return self.head(combined)

    return RetentionHead()


class NeuralImportanceScorer:
    """Predict class probabilities and convert them to a retention policy proxy."""

    model_kind = "neural"

    def __init__(self, model_dir: str | Path = DEFAULT_NEURAL_DIR, *, embedder=None) -> None:
        try:
            import onnxruntime as ort
            from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException
        except ImportError as exc:
            raise ValueError("Install requirements.txt to use the neural scorer") from exc
        directory = Path(model_dir)
        try:
            metadata = json.loads((directory / "metadata.json").read_text())
            if metadata["architecture"] not in {"frozen-bge-two-branch-v1", "frozen-bge-deep-head-v1"}:
                raise ValueError("Incompatible neural architecture")
            if metadata["feature_names"] != list(TEXT_FEATURES):
                raise ValueError("Incompatible neural feature schema")
            if metadata["class_order"] != ["invalid", "short_term", "long_term"]:
                raise ValueError("Incompatible neural class order")
            if metadata["score_values"] != SCORE_VALUES.tolist():
                raise ValueError("Incompatible neural score mapping")
            self.mean = np.asarray(metadata["numeric_mean"], dtype=np.float32)
            self.scale = np.asarray(metadata["numeric_scale"], dtype=np.float32)
            self.temperature = float(metadata["temperature"])
            if (self.mean.shape != (len(TEXT_FEATURES),) or
                    self.scale.shape != self.mean.shape or
                    not np.isfinite(self.mean).all() or
                    not np.isfinite(self.scale).all() or
                    np.any(self.scale <= 0) or not math.isfinite(self.temperature) or
                    self.temperature <= 0):
                raise ValueError("Invalid neural normalization or calibration")
            model_path = directory / "model.onnx"
            if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["model_sha256"]:
                raise ValueError("Neural model checksum mismatch")
            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            if ([item.name for item in session.get_inputs()] != ["embedding", "numeric"] or
                    [item.name for item in session.get_outputs()] != ["logits"]):
                raise ValueError("Incompatible neural ONNX interface")
        except (OSError, KeyError, json.JSONDecodeError, RuntimeError, RuntimeException) as exc:
            raise ValueError(f"Cannot load neural importance model from {directory}: {type(exc).__name__}") from exc
        self.metadata = metadata
        self.session = session
        self._embedder = embedder

    @property
    def embedder(self):
        if self._embedder is None:
            from .retrieval.embeddings import FastEmbedder
            self._embedder = FastEmbedder()
        if self._embedder.dimension != 384 or self._embedder.model_id != self.metadata["encoder_id"]:
            raise ValueError("Incompatible neural text encoder")
        return self._embedder

    def predict_proba(self, content: str, features: dict[str, float]) -> np.ndarray:
        if not content or not content.strip():
            raise ValueError("Neural scorer requires nonempty memory text")
        try:
            numeric = np.asarray([features[name] for name in TEXT_FEATURES], dtype=np.float32)
        except KeyError as exc:
            raise ValueError("Neural scorer requires MLFeatureExtractor features") from exc
        if not np.isfinite(numeric).all():
            raise ValueError("Neural features must be finite")
        embedding = np.asarray(self.embedder.passages([content])[0], dtype=np.float32)
        if embedding.shape != (384,) or not np.isfinite(embedding).all():
            raise ValueError("Neural encoder returned invalid embedding")
        logits = self.session.run(["logits"], {
            "embedding": embedding[None, :],
            "numeric": ((numeric - self.mean) / self.scale)[None, :].astype(np.float32),
        })[0][0] / self.temperature
        shifted = logits - logits.max()
        exponent = np.exp(shifted)
        probabilities = exponent / exponent.sum()
        if (probabilities.shape != (3,) or not np.isfinite(probabilities).all() or
                np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-5)):
            raise ValueError("Neural model returned invalid probabilities")
        return probabilities

    def score_text(self, content: str, features: dict[str, float]) -> float:
        probabilities = self.predict_proba(content, features)
        return round(float(np.dot(probabilities, SCORE_VALUES)), 4)

    def score(self, features: dict[str, float]) -> float:
        """Keep the shared scorer surface explicit about its text requirement."""
        raise ValueError("Neural scorer requires memory text; use score_text")
