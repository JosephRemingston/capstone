"""Stage 1: the ensemble's XGBoost member. xgboost only; never imports torch.

    python -m training.ensemble.xgboost_stage --out artifacts/ensemble_training

Searches max_depth × min_child_weight on validation (early stopping on validation), keeps the winner trained
on the fit split only, and saves it cut to best_iteration + 1 rounds as ``xgboost/xgboost.ubj``.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier

from training.ensemble import datasets, features
from training.ensemble.metrics import classification_metrics, full_metrics

GRID = ((3, 3), (3, 8), (5, 3), (5, 8))   # max_depth x min_child_weight
SEED = 42
PARAMS = dict(objective="multi:softprob", num_class=3, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
              reg_lambda=2.0, tree_method="hist", random_state=SEED, eval_metric="mlogloss")


def sample_weights(labels: np.ndarray) -> np.ndarray:
    """n / (3 * count): balanced class weights."""
    counts = Counter(labels.tolist())
    return np.asarray([len(labels) / (3 * counts[int(label)]) for label in labels])


def make_model(*, max_depth: int, min_child_weight: int, n_estimators: int = 400, n_jobs: int = 4,
               early_stopping_rounds: int | None = 25) -> XGBClassifier:
    options = dict(PARAMS, n_estimators=n_estimators, max_depth=max_depth, min_child_weight=min_child_weight, n_jobs=n_jobs)
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


def sliced_booster(model: XGBClassifier, rounds: int) -> xgb.Booster:
    """The first ``rounds`` boosting rounds: Booster.inplace_predict ignores best_iteration, so cut the model."""
    booster = model.get_booster()[:rounds]
    if booster.num_boosted_rounds() != rounds:
        raise AssertionError("XGBoost slice kept the wrong number of rounds")
    return booster


def run(out_dir: Path, *, n_jobs: int = 4) -> dict:
    started = time.perf_counter()
    out_dir = Path(out_dir)
    dataset, rows, labels, splits = datasets.load_stage0(out_dir)
    class_names = dataset["class_names"]
    folder = out_dir / "xgboost"
    folder.mkdir(parents=True, exist_ok=True)
    fit_idx, val_idx, test_idx = splits["fit"], splits["val"], splits["test"]
    x = features.xgboost_features([row["text"] for row in rows])
    y = labels.astype(np.int32)

    trials, chosen, model = fit_grid(x, y, fit_idx, val_idx, class_names=class_names, n_jobs=n_jobs)
    probs_val = model.predict_proba(x[val_idx]).astype(np.float32)
    probs_test = model.predict_proba(x[test_idx]).astype(np.float32)
    rounds = chosen["best_iteration"] + 1
    booster = sliced_booster(model, rounds)
    booster.save_model(folder / "xgboost.ubj")
    reloaded = xgb.Booster()
    reloaded.load_model(folder / "xgboost.ubj")
    reloaded.set_param({"nthread": 1})
    for name, idx, expected in (("val", val_idx, probs_val), ("test", test_idx, probs_test)):
        if not np.array_equal(np.asarray(reloaded.inplace_predict(x[idx]), dtype=np.float32), expected):
            raise AssertionError(f"saved XGBoost member does not reproduce the {name} probabilities")

    np.save(folder / "probs_val.npy", probs_val)
    np.save(folder / "probs_test.npy", probs_test)
    np.save(folder / "y_val.npy", labels[val_idx])
    np.save(folder / "y_test.npy", labels[test_idx])
    chosen_summary = {key: chosen[key] for key in ("max_depth", "min_child_weight", "best_iteration")}
    result = {"status": "ok", "class_names": class_names, "seconds": round(time.perf_counter() - started, 1),
              "xgboost_version": version("xgboost"), "n_jobs": n_jobs, "feature_count": int(x.shape[1]),
              "params": {**PARAMS, "n_estimators": 400, "early_stopping_rounds": 25}, "trials": trials,
              "chosen": chosen_summary, "rounds": rounds, "model_file": "xgboost.ubj",
              "val": full_metrics(labels[val_idx], probs_val, class_names),
              "test": full_metrics(labels[test_idx], probs_test, class_names)}
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=datasets.DEFAULT_OUT)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args(argv)
    result = run(args.out, n_jobs=args.n_jobs)
    print(json.dumps({"chosen": result["chosen"], "rounds": result["rounds"],
                      "test_macro_f1": result["test"]["macro_f1"], "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
