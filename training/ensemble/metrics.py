"""3-class metrics shared by every stage (sklearn + numpy only)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_recall_fscore_support)

N_CLASSES = 3


def classification_metrics(y, pred, *, class_names) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=[0, 1, 2], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                             "f1": float(f1[i]), "support": int(support[i])}
                      for i, name in enumerate(class_names)},
    }


def nll(y, probs) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)   # float32 softprob rows miss 1 by ~1e-7, which log_loss flags
    return float(log_loss(np.asarray(y), probs, labels=[0, 1, 2]))


def ece(labels, probs, bins=10) -> float:
    """Expected calibration error with equal-width confidence bins."""
    labels, probs = np.asarray(labels), np.asarray(probs)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    return float(sum((mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
                     for i in range(bins)
                     if (mask := (confidence >= edges[i]) &
                         (confidence <= edges[i + 1] if i == bins - 1
                          else confidence < edges[i + 1])).any()))


def full_metrics(y, probs, class_names) -> dict:
    """Argmax classification metrics plus nll / ece_10_bins."""
    probs = np.asarray(probs, dtype=np.float64)
    result = classification_metrics(y, probs.argmax(axis=1), class_names=class_names)
    result["nll"] = nll(y, probs)
    result["ece_10_bins"] = ece(y, probs)
    return result


def majority_class(y) -> int:
    return int(np.argmax(np.bincount(np.asarray(y), minlength=N_CLASSES)))


def confusion_counts(y, pred) -> np.ndarray:
    """Flattened 3x3 confusion counts (gold-major) for one prediction vector or a (B, n) resample matrix."""
    cells = np.asarray(y) * N_CLASSES + np.asarray(pred)
    if cells.ndim == 1:
        return np.bincount(cells, minlength=N_CLASSES ** 2)
    offsets = np.arange(cells.shape[0])[:, None] * N_CLASSES ** 2
    return np.bincount((cells + offsets).ravel(), minlength=N_CLASSES ** 2 * cells.shape[0]).reshape(cells.shape[0], -1)


def macro_f1_from_counts(counts, pred=None) -> np.ndarray | float:
    """Macro-F1 from flattened confusion counts (last axis = 9), or from (y, pred) when pred is given.

    Like sklearn's f1_score(average="macro") without ``labels``: the mean runs over the classes present in
    y or pred (a class absent from both is skipped, not scored 0). Vectorised over leading axes for bootstraps.
    """
    if pred is not None:
        counts = confusion_counts(counts, pred)
    counts = np.asarray(counts, dtype=np.float64)
    matrix = counts.reshape(*counts.shape[:-1], N_CLASSES, N_CLASSES)
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    fn = matrix.sum(axis=-1) - tp
    fp = matrix.sum(axis=-2) - tp
    denominator = 2 * tp + fp + fn
    present = denominator > 0
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(denominator), where=present)
    result = f1.sum(axis=-1) / np.maximum(present.sum(axis=-1), 1)
    return float(result) if result.ndim == 0 else result


def macro_f1(y, pred) -> float:
    return float(macro_f1_from_counts(confusion_counts(y, pred)))
