"""Stage 3: tune the ensemble on validation, score test once, write the serving artifact and the report.

    python -m training.ensemble.assemble --out artifacts/ensemble_training

No torch here: it reads the stage-1/2 outputs, tunes every temperature, the blend weight and the class
weights on validation only, evaluates the class-weighted ensemble on test with bootstrap CIs, then writes
``artifacts/importance_ensemble/`` and loads it back through ``main.ensemble.EnsembleModel`` (xgboost +
onnxruntime) to check that serving reproduces the training predictions, including with live single-text
BGE embeddings as production computes them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

from main.ensemble import (CLASS_ORDER, DEFAULT_ENSEMBLE_DIR, EMBEDDING_DIM, EPS, FORMAT_VERSION, EnsembleModel,
                           apply_class_weights, average, blend, default_embedder, log_probs, model_inputs, softmax)
from main.ml import FEATURE_NAMES, FEATURE_VERSION, TEXT_FEATURES
from training.ensemble import report
from training.ensemble.datasets import DEFAULT_OUT, load_stage0, oracle_predictions
from training.ensemble.features import cached_embeddings
from training.ensemble.heads import ARCHITECTURE
from training.ensemble.metrics import (classification_metrics, confusion_counts, ece, macro_f1, macro_f1_from_counts,
                                       nll)

BLEND_GRID = (0.25, 0.5, 0.75)
CLASS_WEIGHT_GRID = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0)
TEMPERATURE_BOUNDS = (math.log(0.25), math.log(10.0))
RESAMPLES = 1000
BOOTSTRAP_SEED = 42
SEEDS = (42, 43, 44)
GOLDEN_TOLERANCE = 1e-3
GOLDEN_TEXTS = ("I prefer morning meetings.", "My name is Joseph.", "I went to the office yesterday.",
                "Remind me to back up the laptop every Friday.", "Never include peanuts in my meals.",
                "The weather is nice today.", "hello", "Sure, here is a summary of the article you asked about.")


# ----------------------------------------------------------------------------- tuning (validation only)
def fit_temperature(val_logits, val_labels) -> float:
    """Bounded validation-NLL minimisation over log T in [log 0.25, log 10]; float32 logits keep their dtype."""
    val_logits, val_labels = np.asarray(val_logits), np.asarray(val_labels)
    if not np.issubdtype(val_logits.dtype, np.floating):
        val_logits = val_logits.astype(np.float64)

    def nll_at(log_temperature):
        temperature = math.exp(log_temperature)
        shifted = val_logits / temperature
        shifted -= shifted.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        probs = exponent / exponent.sum(axis=1, keepdims=True)
        return log_loss(val_labels, probs, labels=[0, 1, 2])

    calibrated = minimize_scalar(nll_at, bounds=TEMPERATURE_BOUNDS, method="bounded")
    return float(math.exp(calibrated.x))


def search_blend_weight(xgb_val, neural_val, y_val, *, grid=BLEND_GRID) -> tuple[float, list[dict]]:
    """w maximising (val macro-F1, -val NLL); exact ties -> 0.5 when it is in the grid, else grid order."""
    table, best = [], None
    for w in grid:
        probs = blend(xgb_val, neural_val, w)
        entry = {"w": float(w), "macro_f1": macro_f1(y_val, probs.argmax(axis=1)), "nll": nll(y_val, probs)}
        table.append(entry)
        key = (round(entry["macro_f1"], 12), round(-entry["nll"], 12))
        if best is None or key > best[0]:
            best = (key, [w])
        elif key == best[0]:
            best[1].append(w)
    chosen = 0.5 if 0.5 in best[1] else best[1][0]
    return float(chosen), table


def search_class_weights(val_probs, y_val, *, grid=CLASS_WEIGHT_GRID) -> tuple[tuple[float, float, float], dict]:
    """81-point grid over (w0, w1) with w2 = 1; key (macro-F1, balanced acc, -sum|log w|), remaining ties -> grid order."""
    y_val = np.asarray(y_val)
    best = None
    for w0 in grid:
        for w1 in grid:
            weights = (float(w0), float(w1), 1.0)
            pred = apply_class_weights(val_probs, weights)
            counts = confusion_counts(y_val, pred).reshape(3, 3)
            recall = np.divide(np.diag(counts), counts.sum(axis=1), out=np.zeros(3), where=counts.sum(axis=1) > 0)
            key = (round(float(macro_f1_from_counts(counts.ravel())), 12),
                   round(float(recall[counts.sum(axis=1) > 0].mean()), 12),
                   -round(abs(math.log(w0)) + abs(math.log(w1)), 12))
            if best is None or key > best[0]:
                best = (key, weights)
    key, weights = best
    untuned = macro_f1(y_val, np.asarray(val_probs).argmax(axis=1))
    tuned = macro_f1(y_val, apply_class_weights(val_probs, weights))   # exact value; the key is rounded for tie-breaking
    assert tuned + 1e-12 >= untuned, "identity is in the grid, so tuned val F1 cannot drop"
    return weights, {"class_weights": list(weights), "val_macro_f1_before": untuned,
                     "val_macro_f1_after": tuned, "val_balanced_accuracy_after": key[1]}


# ----------------------------------------------------------------------------- bootstraps
def bootstrap_indices(n: int, *, resamples: int = RESAMPLES, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def bootstrap_macro_f1(y, pred, indices) -> np.ndarray:
    y, pred = np.asarray(y), np.asarray(pred)
    return macro_f1_from_counts(confusion_counts(y[indices], pred[indices]))


def percentile_ci(values) -> list[float]:
    low, high = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])
    return [float(low), float(high)]


def sentence_confusions(y, pred, group_ids) -> np.ndarray:
    """Per-sentence flattened confusion counts, (G, 9); resampling sentences = reweighting these rows."""
    y, pred = np.asarray(y), np.asarray(pred)
    _, inverse = np.unique(np.asarray(group_ids), return_inverse=True)
    counts = np.zeros((inverse.max() + 1, 9), dtype=np.int64)
    np.add.at(counts, (inverse, y * 3 + pred), 1)
    return counts


def sentence_bootstrap_macro_f1(y, pred, group_ids, indices) -> np.ndarray:
    """indices: (resamples, G) draws of sentence positions (np.unique order of group_ids)."""
    per_sentence = sentence_confusions(y, pred, group_ids)
    weights = np.zeros((indices.shape[0], per_sentence.shape[0]), dtype=np.int64)
    np.add.at(weights, (np.arange(indices.shape[0])[:, None], indices), 1)
    return macro_f1_from_counts(weights @ per_sentence)


def sentence_level_metrics(y, pred, group_ids, class_names) -> dict:
    """One row per test sentence: label = test-majority class, prediction = majority prediction."""
    labels = oracle_predictions(group_ids, y)
    preds = oracle_predictions(group_ids, pred)
    _, first = np.unique(np.asarray(group_ids), return_index=True)
    return classification_metrics(labels[first], preds[first], class_names=class_names)


# ----------------------------------------------------------------------------- stage-1/2 outputs
def _probs(path: Path, n: int) -> np.ndarray:
    array = np.asarray(np.load(path), dtype=np.float64)
    if array.shape != (n, 3) or not np.isfinite(array).all() or not np.allclose(array.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(f"{path}: expected finite ({n}, 3) probabilities")
    return array


def _logits(path: Path, n: int) -> np.ndarray:
    array = np.load(path)   # float32 kept, so fit_temperature sees the trainer's numbers
    if array.shape != (n, 3) or not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError(f"{path}: expected finite ({n}, 3) float logits")
    return array


def _same_labels(path: Path, expected) -> None:
    if not np.array_equal(np.load(path), expected):
        raise AssertionError(f"{path} differs from the stage-0 labels")


def load_sources(out_dir: Path, labels, splits) -> dict:
    y_val, y_test = labels[splits["val"]], labels[splits["test"]]
    xgb_dir, neural_dir = out_dir / "xgboost", out_dir / "neural"
    for folder in (xgb_dir, neural_dir):
        _same_labels(folder / "y_val.npy", y_val)
        _same_labels(folder / "y_test.npy", y_test)
    seeds = {}
    for seed in SEEDS:
        seed_dir = neural_dir / f"seed_{seed}"
        seeds[seed] = {"info": json.loads((seed_dir / "info.json").read_text()),
                       "logits_val": _logits(seed_dir / "logits_val.npy", len(y_val)),
                       "logits_test": _logits(seed_dir / "logits_test.npy", len(y_test)),
                       "onnx": seed_dir / "model.onnx"}
    return {"xgb": {"probs_val": _probs(xgb_dir / "probs_val.npy", len(y_val)),
                    "probs_test": _probs(xgb_dir / "probs_test.npy", len(y_test)),
                    "result": json.loads((xgb_dir / "result.json").read_text()), "model": xgb_dir / "xgboost.ubj"},
            "neural": {"seeds": seeds, "result": json.loads((neural_dir / "result.json").read_text()),
                       "norm": json.loads((neural_dir / "norm.json").read_text())}}


def tune(sources: dict, y_val) -> dict:
    """Every tuned quantity, chosen on validation only; returns the tuned values and the val/test probabilities."""
    seeds = sources["neural"]["seeds"]
    per_seed = {seed: fit_temperature(seeds[seed]["logits_val"], y_val) for seed in SEEDS}
    val_avg = average([softmax(seeds[s]["logits_val"], per_seed[s]) for s in SEEDS])
    test_avg = average([softmax(seeds[s]["logits_test"], per_seed[s]) for s in SEEDS])
    ensemble_t = fit_temperature(log_probs(val_avg), y_val)
    deep_val, deep_test = softmax(log_probs(val_avg), ensemble_t), softmax(log_probs(test_avg), ensemble_t)
    xgb = sources["xgb"]
    xgb_t = fit_temperature(log_probs(xgb["probs_val"]), y_val)
    xgb_val, xgb_test = softmax(log_probs(xgb["probs_val"]), xgb_t), softmax(log_probs(xgb["probs_test"]), xgb_t)
    w, blend_table = search_blend_weight(xgb_val, deep_val, y_val)
    probs_val, probs_test = blend(xgb_val, deep_val, w), blend(xgb_test, deep_test, w)
    class_weights, threshold_info = search_class_weights(probs_val, y_val)
    return {"per_seed_temperature": per_seed, "ensemble_temperature": ensemble_t, "xgb_temperature": xgb_t,
            "w": w, "blend_table": blend_table, "class_weights": class_weights, "threshold": threshold_info,
            "probs_val": probs_val, "probs_test": probs_test,
            "pred_val": apply_class_weights(probs_val, class_weights),
            "pred_test": apply_class_weights(probs_test, class_weights)}


def class_values(rows: list[dict], labels, splits) -> list[float]:
    """Mean csv5k importance score of the fit+val rows in each class (the serving score of that class)."""
    train = np.concatenate([splits["fit"], splits["val"]])
    scores = np.asarray([rows[i]["score"] for i in train], dtype=np.float64)
    return [round(float(scores[labels[train] == k].mean()), 4) for k in range(3)]


# ----------------------------------------------------------------------------- serving artifact
def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_artifact(artifact_dir: Path, *, dataset: dict, sources: dict, tuned: dict, values: list[float],
                   reference: dict) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    xgb_result = sources["xgb"]["result"]
    shutil.copyfile(sources["xgb"]["model"], artifact_dir / "xgboost.ubj")
    members = []
    for seed in SEEDS:
        info = sources["neural"]["seeds"][seed]["info"]
        name = f"deep_seed_{seed}.onnx"
        shutil.copyfile(sources["neural"]["seeds"][seed]["onnx"], artifact_dir / name)
        members.append({"seed": seed, "file": name, "temperature": tuned["per_seed_temperature"][seed],
                        "best_epoch": info["best_epoch"], "validation_macro_f1": info["validation_macro_f1"],
                        "weights_sha256": info["weights_sha256"]})
    files = ["xgboost.ubj"] + [m["file"] for m in members]
    norm = sources["neural"]["norm"]
    metadata = {
        "format_version": FORMAT_VERSION,
        "model": "csv5k Ensemble + thresholds: fit-only XGBoost + frozen-BGE deep head (3-seed average), class-weighted",
        "target": "cognimem_importance_bins_v1",
        "dataset": {"name": dataset["name"], "sha256": dataset["source"]["sha256"], "n_rows": dataset["n_rows"],
                    "split": dataset["split_protocol"], "seed": dataset["seed"], "split_sizes": dataset["split_sizes"],
                    "bin_edges": dataset["bin_edges"]},
        "class_order": list(CLASS_ORDER), "class_values": values,
        "class_values_definition": "mean csv5k importance score of the fit+val rows in each class",
        "feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES), "text_feature_names": list(TEXT_FEATURES),
        "preprocessing": ("main.ensemble.model_inputs: MemoryClassifier(legacy=True) + MLFeatureExtractor on the text; "
                          "the deep head reads the redact_sensitive text"),
        "files": {name: _sha256(artifact_dir / name) for name in files},
        "xgboost": {"file": "xgboost.ubj", "rounds": xgb_result["rounds"], "best_iteration": xgb_result["chosen"]["best_iteration"],
                    "num_features": len(FEATURE_NAMES), "max_depth": xgb_result["chosen"]["max_depth"],
                    "min_child_weight": xgb_result["chosen"]["min_child_weight"], "params": xgb_result["params"],
                    "temperature": tuned["xgb_temperature"], "xgboost_version": xgb_result["xgboost_version"]},
        "neural": {"architecture": ARCHITECTURE, "encoder_id": sources["neural"]["result"]["encoder_id"],
                   "embedding_dim": EMBEDDING_DIM, "numeric_mean": norm["numeric_mean"], "numeric_scale": norm["numeric_scale"],
                   "onnx_opset": sources["neural"]["result"]["onnx_opset"], "members": members,
                   "combine": "mean over seeds of softmax(logits / T_seed), then softmax(log(mean) / ensemble_temperature)",
                   "ensemble_temperature": tuned["ensemble_temperature"],
                   "torch_version": sources["neural"]["result"]["torch_version"]},
        "blend": {"w_xgb": tuned["w"], "grid": list(BLEND_GRID), "validation_table": tuned["blend_table"]},
        "decision": {"class_weights": list(tuned["class_weights"]), "rule": "argmax(class_weights * p); ties -> lowest index",
                     "grid": list(CLASS_WEIGHT_GRID)},
        "log_clip_eps": EPS,
        "reference": reference,
        "golden": None,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def check_serving(artifact_dir: Path, rows: list[dict], splits: dict, embed_cache: Path, tuned: dict) -> dict:
    """Load the artifact as serving does and compare it with the training predictions."""
    model = EnsembleModel(artifact_dir)   # the constructor, not the per-process cache
    checks = {}
    all_matrix, all_redacted, all_numeric = model_inputs([row["text"] for row in rows])
    all_embeddings = cached_embeddings(all_redacted, embed_cache)   # the exact batch embeddings the heads trained on
    for part in ("val", "test"):
        idx = splits[part]
        probs = model.probabilities(all_matrix[idx], all_embeddings[idx], all_numeric[idx])
        expected = tuned[f"probs_{part}"]
        mismatches = int((model.classes(probs) != tuned[f"pred_{part}"]).sum())
        checks[f"onnx_{part}"] = {"class_mismatches": mismatches, "max_abs_probability_diff": float(np.abs(probs - expected).max())}
        if mismatches:
            raise AssertionError(f"serving model changes {mismatches} {part} predictions")
    # live: one text at a time with the scorer's own encoder, exactly as MemoryCore.process scores a memory
    embedder = default_embedder()
    texts = [rows[i]["text"] for i in splits["test"]]
    matrix, redacted, numeric = model_inputs(texts)
    embeddings = np.asarray([embedder.passages([text])[0] for text in redacted], dtype=np.float32)
    probs = model.probabilities(matrix, embeddings, numeric)
    mismatches = int((model.classes(probs) != tuned["pred_test"]).sum())
    difference = float(np.abs(probs - tuned["probs_test"]).max())
    checks["live_test"] = {"rows": len(texts), "class_mismatches": mismatches, "max_abs_probability_diff": difference,
                           "tolerance": GOLDEN_TOLERANCE}
    if mismatches or difference > GOLDEN_TOLERANCE:
        raise AssertionError(f"live serving differs from training: {checks['live_test']}")
    return checks


def add_golden(artifact_dir: Path) -> dict:
    """Golden texts scored by the serving path; tests compare against them within GOLDEN_TOLERANCE."""
    model = EnsembleModel(artifact_dir)
    embedder = default_embedder()
    vectors = []
    for text in GOLDEN_TEXTS:
        matrix, redacted, numeric = model_inputs([text])
        probs = model.probabilities(matrix, np.asarray(embedder.passages(redacted), dtype=np.float32), numeric)
        vectors.append({"text": text, "class": CLASS_ORDER[int(model.classes(probs)[0])],
                        "probabilities": [float(p) for p in probs[0]]})
    path = artifact_dir / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["golden"] = {"tolerance": GOLDEN_TOLERANCE, "vectors": vectors}
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    EnsembleModel(artifact_dir)   # reload: checksums and schema still valid
    return metadata["golden"]


# ----------------------------------------------------------------------------- driver
def run(out_dir: Path, *, artifact_dir: Path = DEFAULT_ENSEMBLE_DIR, report_dir: Path = report.REPORT_DIR,
        resamples: int = RESAMPLES) -> dict:
    out_dir = Path(out_dir)
    dataset, rows, labels, splits = load_stage0(out_dir)
    class_names = dataset["class_names"]
    y_val, y_test = labels[splits["val"]], labels[splits["test"]]
    group_test = splits["group_ids"][splits["test"]]
    sources = load_sources(out_dir, labels, splits)
    tuned = tune(sources, y_val)

    row_indices = bootstrap_indices(len(y_test), resamples=resamples)
    sentence_indices = bootstrap_indices(len(np.unique(group_test)), resamples=resamples)

    def evaluate(pred) -> dict:
        return {"test": classification_metrics(y_test, pred, class_names=class_names),
                "row_ci95": percentile_ci(bootstrap_macro_f1(y_test, pred, row_indices)),
                "sentence_ci95": percentile_ci(sentence_bootstrap_macro_f1(y_test, pred, group_test, sentence_indices)),
                "sentence_level": sentence_level_metrics(y_test, pred, group_test, class_names)}

    majority = dataset["diagnostics"]["majority"]["class_index"]
    model_result = evaluate(tuned["pred_test"])
    model_result["validation_macro_f1"] = tuned["threshold"]["val_macro_f1_after"]
    model_result["probability_calibration"] = {"test_nll": nll(y_test, tuned["probs_test"]),
                                               "test_ece_10_bins": ece(y_test, tuned["probs_test"]),
                                               "note": "blended probabilities before the class weights"}
    values = class_values(rows, labels, splits)
    reference = {"validation_macro_f1": model_result["validation_macro_f1"],
                 "test_macro_f1": model_result["test"]["macro_f1"], "test_accuracy": model_result["test"]["accuracy"]}
    metadata = write_artifact(Path(artifact_dir), dataset=dataset, sources=sources, tuned=tuned, values=values,
                              reference=reference)
    serving = check_serving(Path(artifact_dir), rows, splits, out_dir / "embeddings", tuned)
    golden = add_golden(Path(artifact_dir))
    summary = {
        "model": metadata["model"], "dataset": dataset, "class_names": class_names, "resamples": resamples,
        "bootstrap_seed": BOOTSTRAP_SEED, "class_values": values,
        "tuned": {"per_seed_temperature": {str(k): v for k, v in tuned["per_seed_temperature"].items()},
                  "ensemble_temperature": tuned["ensemble_temperature"], "xgb_temperature": tuned["xgb_temperature"],
                  "w_xgb": tuned["w"], "blend_table": tuned["blend_table"], **tuned["threshold"]},
        "ensemble_thresholds": model_result,
        "majority": evaluate(np.full_like(y_test, majority)) | {"class_name": class_names[majority]},
        "oracle": evaluate(oracle_predictions(group_test, y_test)) | {"definition": dataset["diagnostics"]["oracle"]["definition"]},
        "xgboost": {key: sources["xgb"]["result"][key] for key in ("chosen", "rounds", "trials", "params", "xgboost_version")},
        "neural": {"architecture": ARCHITECTURE, "seeds": sources["neural"]["result"]["seeds"],
                   "torch_version": sources["neural"]["result"]["torch_version"],
                   "encoder_id": sources["neural"]["result"]["encoder_id"]},
        "serving_checks": serving, "golden": golden,
        "artifact": {"dir": str(Path(artifact_dir).relative_to(report.ROOT)) if Path(artifact_dir).is_relative_to(report.ROOT) else str(artifact_dir),
                     "files": metadata["files"], "metadata_sha256": _sha256(Path(artifact_dir) / "metadata.json")},
        "versions": {package: version(package) for package in ("numpy", "scikit-learn", "xgboost", "onnxruntime", "fastembed")},
    }
    report.write(summary, report_dir)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ENSEMBLE_DIR)
    parser.add_argument("--report-dir", type=Path, default=report.REPORT_DIR)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    args = parser.parse_args(argv)
    summary = run(args.out, artifact_dir=args.artifact_dir, report_dir=args.report_dir, resamples=args.resamples)
    result = summary["ensemble_thresholds"]
    print(json.dumps({"test_macro_f1": result["test"]["macro_f1"], "test_accuracy": result["test"]["accuracy"],
                      "sentence_ci95": result["sentence_ci95"], "class_values": summary["class_values"],
                      "class_weights": summary["tuned"]["class_weights"], "serving_checks": summary["serving_checks"]},
                     indent=2))


if __name__ == "__main__":
    main()
