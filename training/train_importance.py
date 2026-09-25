"""Download Hippocorpus, train XGBoost, and evaluate a fixed held-out test set.

Run: python -m training.train_importance
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import random
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from main import MemoryClassifier, MemoryInput, MemoryRecord, ImportanceScorer
from main.ml import FEATURE_NAMES, FEATURE_VERSION, MLFeatureExtractor

SOURCE_URL = "https://download.microsoft.com/download/3/c/3/3c388755-ac68-4858-8343-9acfb33c631d/hippocorpus-u20220112.zip"
SEED = 42
SOURCE_CSV_SHA256 = "10429da5c70eae01ebd2e80d7e2ea44532ffe381191b3cd48a5f824ab6311530"


def groups_for(rows):
    """Connected components across authors, story pairs, and duplicate text."""
    parents = {}
    def root(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    def join(a, b):
        parents[root(a)] = root(b)
    for row in rows:
        key = "id:" + row["AssignmentId"]
        root(key)
        for field in ("recAgnPairId", "recImgPairId"):
            if row[field]:
                join(key, "id:" + row[field])
        if row["WorkerId"]:
            join(key, "worker:" + row["WorkerId"])
        for field in ("story", "summary"):
            text = " ".join(row[field].casefold().split())
            if text:
                join(key, field + ":" + hashlib.sha256(text.encode()).hexdigest())
    return [root("id:" + row["AssignmentId"]) for row in rows]


def split_indices(groups):
    """Reserve the largest component for training; target 15% each val/test."""
    members = defaultdict(list)
    for index, group in enumerate(groups):
        members[group].append(index)
    largest = max(members, key=lambda key: len(members[key]))
    remaining = sorted(key for key in members if key != largest)
    random.Random(SEED).shuffle(remaining)
    splits = {"train": list(members[largest]), "validation": [], "test": []}
    target = round(len(groups) * 0.15)
    for group in remaining:
        destination = "test" if len(splits["test"]) < target else "validation" if len(splits["validation"]) < target else "train"
        splits[destination].extend(members[group])
    if any(len(indices) < 2 for indices in splits.values()):
        raise ValueError("Insufficient independent groups for three splits")
    return {key: np.array(sorted(value)) for key, value in splits.items()}


def metrics(y, prediction):
    prediction = np.clip(prediction, 0, 1)
    correlation = None if np.ptp(prediction) == 0 or np.ptp(y) == 0 else float(spearmanr(y, prediction).statistic)
    return {"mae": float(mean_absolute_error(y, prediction)),
            "rmse": float(mean_squared_error(y, prediction) ** 0.5),
            "r2": float(r2_score(y, prediction)), "spearman": correlation}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/hippocorpus"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/importance"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    archive = args.data_dir / "source.zip"
    if not archive.exists():
        with urllib.request.urlopen(SOURCE_URL, timeout=90) as response:
            payload = response.read()
        temporary = archive.with_suffix(".download")
        temporary.write_bytes(payload)
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as source:
        raw = source.read("hippoCorpusV2.csv")
    if hashlib.sha256(raw).hexdigest() != SOURCE_CSV_SHA256:
        raise ValueError("Unexpected Hippocorpus CSV checksum; review source version before training")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    all_groups = groups_for(rows)  # Include unlabeled rows when finding related stories.
    valid, groups, labels = [], [], []
    for row, group in zip(rows, all_groups):
        try:
            rating = float(row["importance"])
        except ValueError:
            continue
        if not np.isfinite(rating) or rating not in (1, 2, 3, 4, 5) or not row["story"].strip():
            continue
        valid.append(row)
        groups.append(group)
        labels.append((rating - 1) / 4)
    splits = split_indices(groups)
    classifier, extractor = MemoryClassifier(legacy=True), MLFeatureExtractor()
    feature_rows = []
    for row in valid:
        incoming = MemoryInput(content=row["story"], user_id="training", session_id="training")
        record = MemoryRecord.from_input(incoming, classifier.classify(incoming))
        feature_rows.append(extractor.extract(incoming, record))
    X = np.asarray([[features[name] for name in FEATURE_NAMES] for features in feature_rows], dtype=np.float32)
    y = np.asarray(labels)
    train, val, test = (splits[key] for key in ("train", "validation", "test"))
    candidates = []
    for depth in (2, 4):
        for child_weight in (5, 20):
            model = xgb.XGBRegressor(n_estimators=500, max_depth=depth, min_child_weight=child_weight,
                learning_rate=0.03, subsample=0.85, colsample_bytree=0.8, reg_lambda=10,
                objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
                early_stopping_rounds=35, random_state=SEED, n_jobs=2)
            model.fit(X[train], y[train], eval_set=[(X[val], y[val])], verbose=False)
            result = metrics(y[val], model.predict(X[val]))
            print(json.dumps({"depth": depth, "child_weight": child_weight, "validation": result}), flush=True)
            candidates.append((result["rmse"], model, result))
    _, selected, validation_metrics = min(candidates, key=lambda entry: entry[0])
    # Export exactly the validation-selected trees; test data never used in fitting.
    booster = selected.get_booster()[:selected.best_iteration + 1]
    prediction = np.clip(booster.inplace_predict(X[test]), 0, 1)
    mean = float(y[train].mean())
    # Preserve the historical baseline used by the published Hippocorpus run.
    heuristic_scorer = ImportanceScorer()
    heuristic_scorer.weights["category_episodic"] = 0.12
    heuristic = np.asarray([heuristic_scorer.score(features) for features in feature_rows])
    results = {
        "xgboost": metrics(y[test], prediction),
        "training_mean": metrics(y[test], np.full(len(test), mean)),
        "existing_heuristic": metrics(y[test], heuristic[test]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.ubj"
    booster.save_model(model_path)
    metadata = {
        "feature_version": FEATURE_VERSION, "feature_names": FEATURE_NAMES,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "target": "Hippocorpus self-reported personal-event importance; (rating - 1) / 4",
        "source_url": SOURCE_URL, "source_csv_sha256": hashlib.sha256(raw).hexdigest(),
        "seed": SEED, "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_iteration": selected.best_iteration, "parameters": selected.get_params(),
        "test_metrics": results["xgboost"],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report = {
        "source_url": SOURCE_URL, "source_csv_sha256": metadata["source_csv_sha256"],
        "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "rows_total": len(rows), "rows_used": len(valid), "rows_excluded": len(rows)-len(valid),
        "label_counts": dict(Counter(str(row["importance"]) for row in valid)),
        "story_types": dict(Counter(row["memType"] for row in valid)),
        "split_method": "Seed 42; connected author/pair/duplicate-text components; largest component in train; remaining shuffled, targeting 15% test and 15% validation",
        "splits": {name: {"rows": len(indices), "groups": len({groups[i] for i in indices}),
                           "target_mean": float(y[indices].mean())} for name, indices in splits.items()},
        "group_overlap": {"train_validation": len({groups[i] for i in train} & {groups[i] for i in val}),
                          "train_test": len({groups[i] for i in train} & {groups[i] for i in test}),
                          "validation_test": len({groups[i] for i in val} & {groups[i] for i in test})},
        "validation_candidates": [{"max_depth": m.max_depth, "min_child_weight": m.min_child_weight,
                                   "best_iteration": m.best_iteration, "metrics": result} for _,m,result in candidates],
        "selected_validation_metrics": validation_metrics, "test_metrics": results,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__, "xgboost": xgb.__version__},
        "limitations": ["Personal-event importance is a proxy, not conversational retention utility.",
                        "Only text-derived features used; no demographics, survey responses, or target-derived features.",
                        "Author/pair links create a large component reserved for training; test represents smaller components.",
                        "No conversational importance or tier ground truth; lifecycle thresholds remain unvalidated for this model."],
    }
    (args.report_dir / "importance_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.report_dir / "importance_test_predictions.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["story_id", "target", "prediction", "mean_baseline", "heuristic"])
        for i, predicted in zip(test, prediction):
            writer.writerow([valid[i]["AssignmentId"], y[i], float(predicted), mean, heuristic[i]])
    (args.report_dir / "importance_splits.json").write_text(json.dumps({name: [valid[i]["AssignmentId"] for i in indices] for name,indices in splits.items()}, indent=2) + "\n")
    print(json.dumps({"splits": report["splits"], "test_metrics": results}, indent=2))


if __name__ == "__main__":
    main()
