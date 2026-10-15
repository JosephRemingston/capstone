"""Stage 1: the shipped XGBoost retention procedure on one dataset. xgboost only; never imports torch.

    python -m training.experiments.xgboost_run --dataset-dir artifacts/experiments/retention_datasets/msc
    python -m training.experiments.xgboost_run --dataset-dir ... --bin-edges tertile   # sensitivity, same partition
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from training.experiments import datasets, features
from training.experiments.metrics import classification_metrics, full_metrics

GRID = ((3, 3), (3, 8), (5, 3), (5, 8))   # max_depth x min_child_weight, training/train_conversational_retention.py:121-122
SEED = 42


def sample_weights(labels: np.ndarray) -> np.ndarray:
    """n / (3 * count), training/train_conversational_retention.py:116-117."""
    counts = Counter(labels.tolist())
    return np.asarray([len(labels) / (3 * counts[int(label)]) for label in labels])


def make_model(*, max_depth: int, min_child_weight: int, n_estimators: int = 400, seed: int = SEED,
               n_jobs: int = 4, early_stopping_rounds: int | None = 25) -> XGBClassifier:
    """Shipped settings, training/train_conversational_retention.py:123-129 and :144-151."""
    options = dict(objective="multi:softprob", num_class=3, n_estimators=n_estimators, learning_rate=0.05,
                   max_depth=max_depth, min_child_weight=min_child_weight, subsample=0.85,
                   colsample_bytree=0.85, reg_lambda=2.0, tree_method="hist", n_jobs=n_jobs,
                   random_state=seed, eval_metric="mlogloss")
    if early_stopping_rounds is not None:
        options["early_stopping_rounds"] = early_stopping_rounds
    return XGBClassifier(**options)


def fit_grid(x, y, fit_idx, val_idx, *, class_names, n_jobs: int = 4):
    """Returns (trials, chosen trial, chosen fitted model); key (macro_f1, balanced_accuracy, -best_iteration)."""
    weights = sample_weights(y[fit_idx])
    trials, best = [], None
    for depth, child_weight in GRID:
        model = make_model(max_depth=depth, min_child_weight=child_weight, n_jobs=n_jobs)
        model.fit(x[fit_idx], y[fit_idx], sample_weight=weights,
                  eval_set=[(x[val_idx], y[val_idx])], verbose=False)
        result = classification_metrics(y[val_idx], model.predict(x[val_idx]), class_names=class_names)
        trial = {"max_depth": depth, "min_child_weight": child_weight,
                 "best_iteration": int(model.best_iteration), **result}
        trials.append(trial)
        key = (result["macro_f1"], result["balanced_accuracy"], -model.best_iteration)
        if best is None or key > best[0]:
            best = (key, trial, model)
    return trials, best[1], best[2]


def refit(x, y, train_idx, chosen: dict, *, n_jobs: int = 4) -> XGBClassifier:
    """best_iteration + 1 trees on fit + val without early stopping, training/train_conversational_retention.py:141-152."""
    model = make_model(max_depth=chosen["max_depth"], min_child_weight=chosen["min_child_weight"],
                       n_estimators=chosen["best_iteration"] + 1, n_jobs=n_jobs, early_stopping_rounds=None)
    model.fit(x[train_idx], y[train_idx], sample_weight=sample_weights(y[train_idx]))
    return model


def run(dataset_dir: Path, *, bin_edges: str = "primary", n_jobs: int = 4) -> dict:
    started = time.perf_counter()
    dataset_dir = Path(dataset_dir)
    dataset, rows, labels, splits = datasets.load_stage0(dataset_dir)
    if bin_edges == "tertile":
        if "tertile_labels" not in splits:
            raise SystemExit(f"{dataset['name']}: no tertile labels (sensitivity applies to CSV datasets only)")
        labels, folder = splits["tertile_labels"], dataset_dir / "xgboost_tertile"
    else:
        folder = dataset_dir / "xgboost"
    folder.mkdir(parents=True, exist_ok=True)
    class_names = dataset["class_names"]
    fit_idx, val_idx, test_idx = splits["fit"], splits["val"], splits["test"]
    x = features.xgboost_features([row["text"] for row in rows])
    y = labels.astype(np.int32)

    trials, chosen, fit_only = fit_grid(x, y, fit_idx, val_idx, class_names=class_names, n_jobs=n_jobs)
    # original row order: subsample=0.85 draws rows by position, so order changes the trees (the shipped refit used x_train as-is)
    final = refit(x, y, np.sort(np.concatenate([fit_idx, val_idx])), chosen, n_jobs=n_jobs)
    probs_val = fit_only.predict_proba(x[val_idx]).astype(np.float32)
    probs_test_fitonly = fit_only.predict_proba(x[test_idx]).astype(np.float32)
    probs_test_refit = final.predict_proba(x[test_idx]).astype(np.float32)
    proxy = dataset["kind"] == "msc"

    np.save(folder / "probs_val.npy", probs_val)
    np.save(folder / "probs_test_fitonly.npy", probs_test_fitonly)
    np.save(folder / "probs_test_refit.npy", probs_test_refit)
    np.save(folder / "y_val.npy", labels[val_idx])
    np.save(folder / "y_test.npy", labels[test_idx])
    final.save_model(folder / "model_refit.ubj")
    chosen_summary = {key: chosen[key] for key in ("max_depth", "min_child_weight", "best_iteration")}
    (folder / "grid.json").write_text(json.dumps({"trials": trials, "chosen": chosen_summary}, indent=2) + "\n")
    result = {"status": "ok", "dataset": dataset["name"], "bin_edges": bin_edges, "class_names": class_names,
              "seconds": round(time.perf_counter() - started, 1), "xgboost_version": version("xgboost"),
              "n_jobs": n_jobs, "feature_count": int(x.shape[1]), "trials": trials, "chosen": chosen_summary,
              "refit_n_estimators": chosen["best_iteration"] + 1,
              "val": full_metrics(labels[val_idx], probs_val, class_names),
              "test_refit": full_metrics(labels[test_idx], probs_test_refit, class_names, include_proxy=proxy),
              "test_fitonly": full_metrics(labels[test_idx], probs_test_fitonly, class_names, include_proxy=proxy)}
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--bin-edges", choices=("primary", "tertile"), default="primary")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args(argv)
    result = run(args.dataset_dir, bin_edges=args.bin_edges, n_jobs=args.n_jobs)
    print(json.dumps({"dataset": result["dataset"], "bin_edges": result["bin_edges"], "chosen": result["chosen"],
                      "test_refit_macro_f1": result["test_refit"]["macro_f1"],
                      "test_fitonly_macro_f1": result["test_fitonly"]["macro_f1"], "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
