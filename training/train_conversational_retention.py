"""Train a conversational retention proxy from human-labeled MSC facts.

The source has validity and duration labels, not numeric importance ratings.
Class 0 is invalid, 1 short-term, and 2 long-term. The optional scorer maps
their predicted probabilities to 0, 0.5, and 1 respectively. This mapping is
an explicit policy proxy and must not be reported as gold importance.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import f1_score, mean_absolute_error, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from main.domain.classification import MemoryClassifier
from main.domain.models import MemoryInput, MemoryRecord
from main.ml import FEATURE_NAMES, FEATURE_VERSION, MLFeatureExtractor


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "personal_facts_msc"
MODEL_DIR = ROOT / "artifacts" / "importance"
REPORT_DIR = ROOT / "evaluation" / "reports"
SOURCE = "https://huggingface.co/datasets/adugeen/personal-facts-msc"
FILES = {
    "train": ("f7f7fab70fcfc949abc743480fbf189d5956fc2d05cbd0628147aa76d1551b8b", "train-00000-of-00001.parquet"),
    "test": ("91c4c166f63aa6b747604774a04d01c5b4fd5a8deedca88797a339e087640aaa", "test-00000-of-00001.parquet"),
}
CLASSES = ("invalid", "short_term", "long_term")
SCORE_VALUES = (0.0, 0.5, 1.0)
SEED = 42


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_rows(split: str) -> list[dict]:
    expected, original_name = FILES[split]
    path = DATA_DIR / f"{split}.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{SOURCE}/resolve/main/data/{original_name}"
        with urlopen(url, timeout=60) as response:
            content = response.read()
        if _sha256(content) != expected:
            raise ValueError(f"Unexpected checksum for {split} dataset")
        path.write_bytes(content)
    if _sha256(path.read_bytes()) != expected:
        raise ValueError(f"Unexpected checksum for {split} dataset")
    return pq.read_table(path).to_pylist()


def target(row: dict) -> int | None:
    if row["broken"] == "Yes":
        return 0
    if row["broken"] == "No" and row["duration"] == "Short-term":
        return 1
    if row["broken"] == "No" and row["duration"] == "Long-term":
        return 2
    return None


def prepare(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, int]:
    classifier = MemoryClassifier(legacy=True)
    extractor = MLFeatureExtractor()
    matrix: list[list[float]] = []
    labels: list[int] = []
    omitted = 0
    for row in rows:
        label = target(row)
        if label is None or not row["text"].strip():
            omitted += 1
            continue
        incoming = MemoryInput(content=row["text"], user_id="dataset", session_id="dataset")
        record = MemoryRecord.from_input(incoming, classifier.classify(incoming))
        features = extractor.extract(incoming, record)
        matrix.append([features[name] for name in FEATURE_NAMES])
        labels.append(label)
    return np.asarray(matrix, dtype=np.float32), np.asarray(labels, dtype=np.int32), omitted


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
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
                      for i, name in enumerate(CLASSES)},
    }


def run() -> dict:
    train_rows = load_rows("train")
    test_rows = load_rows("test")
    x_train, y_train, omitted_train = prepare(train_rows)
    x_test, y_test, omitted_test = prepare(test_rows)
    fit_idx, val_idx = train_test_split(
        np.arange(len(y_train)), test_size=0.2, stratify=y_train, random_state=SEED
    )
    counts = Counter(y_train[fit_idx].tolist())
    weights = np.asarray([len(fit_idx) / (3 * counts[int(label)]) for label in y_train[fit_idx]])

    trials = []
    best = None
    for depth in (3, 5):
        for child_weight in (3, 8):
            model = XGBClassifier(
                objective="multi:softprob", num_class=3, n_estimators=400,
                learning_rate=0.05, max_depth=depth, min_child_weight=child_weight,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
                tree_method="hist", n_jobs=4, random_state=SEED,
                eval_metric="mlogloss", early_stopping_rounds=25,
            )
            model.fit(x_train[fit_idx], y_train[fit_idx], sample_weight=weights,
                      eval_set=[(x_train[val_idx], y_train[val_idx])], verbose=False)
            predicted = model.predict(x_train[val_idx])
            result = metrics(y_train[val_idx], predicted)
            trial = {"max_depth": depth, "min_child_weight": child_weight,
                     "best_iteration": int(model.best_iteration), **result}
            trials.append(trial)
            key = (result["macro_f1"], result["balanced_accuracy"], -model.best_iteration)
            if best is None or key > best[0]:
                best = (key, trial)

    chosen = best[1]
    full_counts = Counter(y_train.tolist())
    full_weights = np.asarray([len(y_train) / (3 * full_counts[int(label)]) for label in y_train])
    final_model = XGBClassifier(
        objective="multi:softprob", num_class=3,
        n_estimators=chosen["best_iteration"] + 1, learning_rate=0.05,
        max_depth=chosen["max_depth"], min_child_weight=chosen["min_child_weight"],
        subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
        tree_method="hist", n_jobs=4, random_state=SEED,
        eval_metric="mlogloss",
    )
    final_model.fit(x_train, y_train, sample_weight=full_weights)
    probabilities = final_model.predict_proba(x_test)
    predicted = np.argmax(probabilities, axis=1)
    test_metrics = metrics(y_test, predicted)
    proxy_truth = np.asarray(SCORE_VALUES)[y_test]
    proxy_prediction = probabilities @ np.asarray(SCORE_VALUES)
    test_metrics["proxy_mae"] = float(mean_absolute_error(proxy_truth, proxy_prediction))

    majority = int(np.argmax(np.bincount(y_train)))
    majority_metrics = metrics(y_test, np.full_like(y_test, majority))
    report = {
        "dataset": SOURCE,
        "target": "human-annotated fact validity and expected duration; numeric score is a policy proxy",
        "class_order": list(CLASSES), "score_values": list(SCORE_VALUES),
        "dataset_size": {"train": len(train_rows), "test": len(test_rows),
                         "train_used": len(y_train), "test_used": len(y_test),
                         "train_omitted": omitted_train, "test_omitted": omitted_test},
        "train_class_counts": {CLASSES[i]: int((y_train == i).sum()) for i in range(3)},
        "test_class_counts": {CLASSES[i]: int((y_test == i).sum()) for i in range(3)},
        "selection": "20% stratified validation from published train split; maximize macro F1",
        "validation_trials": trials, "selected_hyperparameters": {
            "max_depth": chosen["max_depth"], "min_child_weight": chosen["min_child_weight"],
            "n_estimators": chosen["best_iteration"] + 1},
        "test": test_metrics, "majority_baseline": majority_metrics,
        "limitations": ["No human numeric importance scores", "Single annotator",
                        "English extracted fact candidates; task requests and greetings are underrepresented",
                        "Published split is stratified by category, not by speaker or conversation"],
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "model.ubj"
    final_model.get_booster().save_model(model_path)
    metadata = {
        "model": "xgboost", "target": "conversational_retention_proxy_v1",
        "output_type": "multiclass_probability", "class_order": list(CLASSES),
        "score_values": list(SCORE_VALUES), "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES, "model_sha256": _sha256(model_path.read_bytes()),
        "dataset": SOURCE, "dataset_file_sha256": {k: FILES[k][0] for k in FILES},
        "selection_seed": SEED, "selected_hyperparameters": report["selected_hyperparameters"],
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "conversational_retention.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"test": result["test"], "majority_baseline": result["majority_baseline"],
                      "dataset_size": result["dataset_size"]}, indent=2))
