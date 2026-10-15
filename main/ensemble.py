"""Default importance scorer: 25 % csv5k ensemble + 75 % heuristic.

The ensemble in ``artifacts/importance_ensemble/`` blends a fit-only XGBoost model with three seeds of a
frozen-BGE deep head and applies tuned class weights to pick low / mid / high importance (training:
``python -m training.ensemble.run``). ``HybridImportanceScorer`` maps that class to the mean csv5k importance
of its training rows and mixes it with the heuristic ``ImportanceScorer``.

xgboost, onnxruntime and fastembed load only when a model or encoder is built, so importing this module
is cheap and safe from the torch training process. ``model_inputs`` and the numpy helpers are shared with
training, so serving runs the code the training gates checked.
"""
from __future__ import annotations

import atexit
import functools
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .domain.classification import MemoryClassifier
from .domain.models import MemoryInput, MemoryRecord
from .domain.privacy import redact_sensitive
from .domain.scoring import ImportanceScorer
from .ml import FEATURE_NAMES, FEATURE_VERSION, MLFeatureExtractor, TEXT_FEATURES

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENSEMBLE_DIR = ROOT / "artifacts" / "importance_ensemble"
BGE_CACHE_DIR = ROOT / "data" / "rag" / "models"
FORMAT_VERSION = 1
CLASS_ORDER = ("low", "mid", "high")
EMBEDDING_DIM = 384
MODEL_WEIGHT = 0.25          # user-chosen mix: 25 % model, 75 % heuristic
SMALL_TALK_PENALTY = 0.25    # ImportanceScorer.temporary_penalty, applied to the model part of TEMPORARY records
SCORERS = ("hybrid", "heuristic")
EPS = 1e-12
TRAIN_COMMAND = "python -m training.ensemble.run"
OPT_OUT = "set COGNIMEM_SCORER=heuristic to run without it"


# ----------------------------------------------------------------------------- shared with training
def softmax(logits, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64) / temperature
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def log_probs(probs) -> np.ndarray:
    """Logits for temperature scaling of a model that only exposes probabilities (XGBoost, averages)."""
    return np.log(np.clip(np.asarray(probs, dtype=np.float64), EPS, 1.0))


def average(prob_list) -> np.ndarray:
    return np.mean([np.asarray(p, dtype=np.float64) for p in prob_list], axis=0)


def blend(xgb, neural, w: float) -> np.ndarray:
    return w * np.asarray(xgb, dtype=np.float64) + (1.0 - w) * np.asarray(neural, dtype=np.float64)


def apply_class_weights(probs, weights) -> np.ndarray:
    return np.argmax(np.asarray(probs, dtype=np.float64) * np.asarray(weights, dtype=np.float64), axis=1)


def model_inputs(texts: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(XGBoost matrix (n, 525) float32, redacted texts, numeric (n, 13) float32) for the ensemble.

    The model's own legacy view of the text: a fresh record classified by ``MemoryClassifier(legacy=True)``
    and ``MLFeatureExtractor`` (text-only features). XGBoost reads the text as given; the deep head reads
    the ``redact_sensitive`` version, as in training. The caller's record keeps its own category.
    """
    classifier = MemoryClassifier(legacy=True)
    extractor = MLFeatureExtractor()

    def features(text: str) -> dict[str, float]:
        incoming = MemoryInput(content=text, user_id="model-input", session_id="model-input")
        record = MemoryRecord.from_input(incoming, classifier.classify(incoming))
        return extractor.extract(incoming, record)

    matrix, redacted, numeric = [], [], []
    for text in texts:
        raw = features(text)
        clean, _ = redact_sensitive(text)
        neural = raw if clean == text else features(clean)
        matrix.append([raw[name] for name in FEATURE_NAMES])
        redacted.append(clean)
        numeric.append([neural[name] for name in TEXT_FEATURES])
    return (np.asarray(matrix, dtype=np.float32).reshape(len(texts), len(FEATURE_NAMES)), redacted,
            np.asarray(numeric, dtype=np.float32).reshape(len(texts), len(TEXT_FEATURES)))


# ----------------------------------------------------------------------------- model bundle
def _positive(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def validate_metadata(metadata: dict) -> None:
    """Raise ValueError unless the metadata describes a loadable low/mid/high ensemble."""
    xgb, neural, decision = metadata["xgboost"], metadata["neural"], metadata["decision"]
    values = metadata["class_values"]
    if metadata["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported importance model format")
    if metadata["class_order"] != list(CLASS_ORDER):
        raise ValueError("Incompatible importance class order")
    if (len(values) != 3 or not all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in values)
            or not values[0] < values[1] < values[2]):
        raise ValueError("Importance class values must increase within [0, 1]")
    if metadata["feature_version"] != FEATURE_VERSION or metadata["feature_names"] != FEATURE_NAMES:
        raise ValueError("Incompatible importance feature schema")
    if metadata["text_feature_names"] != list(TEXT_FEATURES):
        raise ValueError("Incompatible importance numeric feature schema")
    if xgb["num_features"] != len(FEATURE_NAMES) or xgb["rounds"] != xgb["best_iteration"] + 1:
        raise ValueError("Incompatible XGBoost member")
    mean = np.asarray(neural["numeric_mean"], dtype=np.float32)
    scale = np.asarray(neural["numeric_scale"], dtype=np.float32)
    if (mean.shape != (len(TEXT_FEATURES),) or scale.shape != mean.shape or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("Invalid numeric normalisation")
    if neural["embedding_dim"] != EMBEDDING_DIM or not neural["members"]:
        raise ValueError("Incompatible deep-head members")
    if not isinstance(neural.get("encoder_id"), str) or not neural["encoder_id"]:
        raise ValueError("The importance model does not name its text encoder")
    temperatures = [xgb["temperature"], neural["ensemble_temperature"]] + [m["temperature"] for m in neural["members"]]
    if not all(_positive(t) for t in temperatures):
        raise ValueError("Temperatures must be finite and positive")
    w = metadata["blend"]["w_xgb"]
    if not isinstance(w, (int, float)) or not 0.0 <= w <= 1.0:
        raise ValueError("Blend weight must lie in [0, 1]")
    if len(decision["class_weights"]) != 3 or not all(_positive(c) for c in decision["class_weights"]):
        raise ValueError("Class weights must be finite and positive")


class EnsembleModel:
    """The validated csv5k ensemble: XGBoost + deep-head seeds + temperatures + class weights."""

    def __init__(self, model_dir: str | Path = DEFAULT_ENSEMBLE_DIR) -> None:
        directory = Path(model_dir)
        if not (directory / "metadata.json").is_file():
            raise ValueError(f"Importance model not found in {directory}; train it with `{TRAIN_COMMAND}` or {OPT_OUT}")
        try:
            import onnxruntime as ort
            import xgboost as xgb
        except ImportError as exc:
            raise ValueError(f"Install requirements.txt to use the importance model, or {OPT_OUT}") from exc
        try:
            raw = (directory / "metadata.json").read_bytes()
            metadata = json.loads(raw)
            validate_metadata(metadata)
            files = {}
            for name, digest in metadata["files"].items():
                data = (directory / name).read_bytes()
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError(f"Importance model checksum mismatch: {name}")
                files[name] = data
            booster = xgb.Booster()
            booster.load_model(bytearray(files[metadata["xgboost"]["file"]]))
            booster.set_param({"nthread": 1})
            if (booster.num_boosted_rounds() != metadata["xgboost"]["rounds"]
                    or booster.num_features() != len(FEATURE_NAMES)):
                raise ValueError("XGBoost member does not match its metadata")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            sessions = []
            for member in metadata["neural"]["members"]:
                session = ort.InferenceSession(files[member["file"]], sess_options=options,
                                               providers=["CPUExecutionProvider"])
                if ([item.name for item in session.get_inputs()] != ["embedding", "numeric"]
                        or [item.name for item in session.get_outputs()] != ["logits"]):
                    raise ValueError("Incompatible deep-head ONNX interface")
                sessions.append(session)
        except ValueError as exc:
            raise ValueError(f"Cannot load importance model from {directory}: {exc}") from exc
        except Exception as exc:   # OSError, KeyError, JSON, XGBoost and onnxruntime errors alike
            raise ValueError(f"Cannot load importance model from {directory}: {type(exc).__name__}: {exc}") from exc
        neural = metadata["neural"]
        self.directory = directory
        self.metadata = metadata
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.encoder_id = neural["encoder_id"]
        self.class_values = tuple(float(v) for v in metadata["class_values"])
        self._booster = booster
        self._sessions = sessions
        self._member_temperatures = [float(m["temperature"]) for m in neural["members"]]
        self._mean = np.asarray(neural["numeric_mean"], dtype=np.float32)
        self._scale = np.asarray(neural["numeric_scale"], dtype=np.float32)
        self._xgb_temperature = float(metadata["xgboost"]["temperature"])
        self._ensemble_temperature = float(neural["ensemble_temperature"])
        self._w = float(metadata["blend"]["w_xgb"])
        self._class_weights = np.asarray(metadata["decision"]["class_weights"], dtype=np.float64)

    def probabilities(self, xgb_matrix, embeddings, numeric) -> np.ndarray:
        """Blended low/mid/high probabilities (n, 3) before the class weights."""
        xgb_matrix = np.asarray(xgb_matrix, dtype=np.float32)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        numeric = np.asarray(numeric, dtype=np.float32)
        n = len(xgb_matrix)
        if (xgb_matrix.shape != (n, len(FEATURE_NAMES)) or embeddings.shape != (n, EMBEDDING_DIM)
                or numeric.shape != (n, len(TEXT_FEATURES))):
            raise ValueError("Importance model inputs have the wrong shape")
        if not (np.isfinite(xgb_matrix).all() and np.isfinite(embeddings).all() and np.isfinite(numeric).all()):
            raise ValueError("Importance model inputs must be finite")
        xgb_probs = np.asarray(self._booster.inplace_predict(xgb_matrix), dtype=np.float32).reshape(n, 3)
        xgb_calibrated = softmax(log_probs(xgb_probs), self._xgb_temperature)
        scaled = ((numeric - self._mean) / self._scale).astype(np.float32)
        seeds = [softmax(session.run(["logits"], {"embedding": embeddings, "numeric": scaled})[0], temperature)
                 for session, temperature in zip(self._sessions, self._member_temperatures)]
        deep = softmax(log_probs(average(seeds)), self._ensemble_temperature)
        probs = blend(xgb_calibrated, deep, self._w)
        if not (np.isfinite(probs).all() and np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)):
            raise ValueError("Importance model returned invalid probabilities")
        return probs

    def classes(self, probs) -> np.ndarray:
        return apply_class_weights(probs, self._class_weights)


@functools.lru_cache(maxsize=4)
def _load_model(resolved_dir: str) -> EnsembleModel:
    return EnsembleModel(resolved_dir)


def load_model(model_dir: str | Path | None = None) -> EnsembleModel:
    """The process-wide model for a directory; runners that build a MemoryCore per row load it once."""
    return _load_model(str(Path(model_dir or DEFAULT_ENSEMBLE_DIR).resolve()))


def local_embedder(threads: int | None = None):
    """A BGE encoder read only from the local cache (never a silent download); threads=None keeps FastEmbedder's default."""
    from .retrieval.embeddings import FastEmbedder
    try:
        return FastEmbedder(cache_dir=BGE_CACHE_DIR, local_files_only=True,
                            **({} if threads is None else {"threads": threads}))
    except Exception as exc:
        if isinstance(exc.__cause__, ImportError):   # fastembed itself is missing: FastEmbedder already says so
            raise
        raise ValueError(
            f"The BGE encoder is not cached in {BGE_CACHE_DIR}; download it once with "
            "`.venv/bin/python -c \"from main.retrieval.embeddings import FastEmbedder; FastEmbedder()\"` "
            f"from the repository root, or {OPT_OUT}") from exc


@functools.lru_cache(maxsize=1)
def default_embedder():
    """The scorer's own encoder, shared per process."""
    return local_embedder()


@atexit.register
def _release_native_models() -> None:
    # Drop the cached onnxruntime sessions, XGBoost booster and encoder while the interpreter is still whole:
    # left to static teardown, their thread pools have occasionally aborted the process at exit.
    _load_model.cache_clear()
    default_embedder.cache_clear()


# ----------------------------------------------------------------------------- scorer
class HybridImportanceScorer:
    """``MODEL_WEIGHT`` × the ensemble's class value + the rest × the heuristic score."""

    kind = "hybrid"

    def __init__(self, model: EnsembleModel, *, embedder=None, heuristic: ImportanceScorer | None = None,
                 model_weight: float = MODEL_WEIGHT) -> None:
        embedder = embedder if embedder is not None else default_embedder()
        if getattr(embedder, "dimension", None) != EMBEDDING_DIM or embedder.model_id != model.encoder_id:
            raise ValueError(f"The text encoder {getattr(embedder, 'model_id', None)!r} does not match the "
                             f"importance model's {model.encoder_id!r}")
        if not 0.0 <= model_weight <= 1.0:
            raise ValueError("model_weight must lie in [0, 1]")
        self.model = model
        self.embedder = embedder
        self.heuristic = heuristic or ImportanceScorer()
        self.model_weight = float(model_weight)

    def predict(self, content: str) -> tuple[str, np.ndarray]:
        """(class name, blended probabilities) for one memory text."""
        if not content or not content.strip():
            raise ValueError("The importance model requires nonempty memory text")
        matrix, redacted, numeric = model_inputs([content])
        embeddings = np.asarray(self.embedder.passages(redacted), dtype=np.float32)
        probs = self.model.probabilities(matrix, embeddings, numeric)
        return CLASS_ORDER[int(self.model.classes(probs)[0])], probs[0]

    def assess(self, content: str, features: dict[str, float]) -> tuple[float, dict]:
        """(score, provenance stamp) for one memory: the core stores both."""
        name, probs = self.predict(content)
        small_talk = features.get("category_temporary", 0.0) >= 1.0
        model_score = self.model.class_values[CLASS_ORDER.index(name)] * (SMALL_TALK_PENALTY if small_talk else 1.0)
        heuristic_score = self.heuristic.score(features)
        score = round(max(0.0, min(self.model_weight * model_score + (1.0 - self.model_weight) * heuristic_score, 1.0)), 4)
        stamp = {"type": self.kind, "model_weight": self.model_weight, "model_class": name,
                 "model_probabilities": [round(float(p), 4) for p in probs], "model_score": round(model_score, 4),
                 "small_talk_penalty": small_talk, "heuristic_score": heuristic_score,
                 "model_sha256": self.model.sha256, "encoder": self.model.encoder_id}
        return score, stamp

    def score_text(self, content: str, features: dict[str, float]) -> float:
        return self.assess(content, features)[0]

    def score(self, features: dict[str, float]) -> float:
        raise ValueError("The hybrid importance scorer requires memory text; use score_text")


def selected_scorer(name: str | None = None) -> str:
    """The scorer name from the argument or COGNIMEM_SCORER (default 'hybrid'); anything else is an error."""
    value = (name if name is not None else os.getenv("COGNIMEM_SCORER") or "").strip().lower() or "hybrid"   # blank = unset
    if value not in SCORERS:
        raise ValueError(f"COGNIMEM_SCORER must be one of {', '.join(SCORERS)}, not {value!r}")
    return value


def build_scorer(name: str | None = None, *, model_dir: str | Path | None = None, embedder=None):
    """A fresh scorer of the selected kind; the model bundle and default encoder are shared per process."""
    if selected_scorer(name) == "heuristic":
        if model_dir is not None:
            raise ValueError("A model directory requires the hybrid scorer")
        return ImportanceScorer()
    return HybridImportanceScorer(load_model(model_dir), embedder=embedder)


def default_importance_scorer():
    return build_scorer()


def describe_scorer(scorer) -> dict:
    """Run-level provenance for evaluation reports: which scorer ran, and on which model file."""
    if isinstance(scorer, HybridImportanceScorer):
        return {"type": scorer.kind, "model_weight": scorer.model_weight, "small_talk_penalty": SMALL_TALK_PENALTY,
                "model_sha256": scorer.model.sha256, "encoder": scorer.model.encoder_id}
    return {"type": "heuristic" if isinstance(scorer, ImportanceScorer) else type(scorer).__name__}
