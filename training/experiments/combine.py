"""Stage 3: seed averaging, ensembles, thresholds, bootstraps and paired deltas (no native ML libs).

    python -m training.experiments.combine --dataset-dir artifacts/experiments/retention_datasets/csv5k [--resamples 1000]

Reads whatever subset of the stage-1 (``xgboost/``, ``xgboost_tertile/``) and stage-2 (``neural/``)
outputs exists under the dataset directory and writes ``<dataset_dir>/combined.json``; variants that
cannot be built are recorded as missing. Every tuned quantity is chosen on validation only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

from evaluation.stats import paired_comparison
from training.experiments.datasets import load_stage0, oracle_predictions
from training.experiments.metrics import (classification_metrics, confusion_counts, full_metrics,
                                          macro_f1, macro_f1_from_counts, nll)

ARCHITECTURES = {"deep": "frozen-bge-deep-head-v1", "shallow": "frozen-bge-two-branch-v1"}
BLEND_GRID = (0.25, 0.5, 0.75)
CLASS_WEIGHT_GRID = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0)
TEMPERATURE_BOUNDS = (math.log(0.25), math.log(10.0))
RESAMPLES = 1000
BOOTSTRAP_SEED = 42
BONFERRONI_ALPHA = 0.0125
VARIANT_ORDER = ("majority", "oracle", "xgb_asis", "xgb_fitonly", "deep_asis", "shallow_asis", "deep_seedavg",
                 "shallow_seedavg", "ensemble", "thr_xgb_fitonly", "thr_deep_asis", "thr_shallow_asis",
                 "thr_ensemble", "xgb_tertile")
# (baseline, candidate): the candidate is the change under test, delta = candidate - baseline
PRIMARY_COMPARISONS = (("deep_asis", "deep_seedavg"), ("deep_seedavg", "ensemble"),
                       ("xgb_fitonly", "ensemble"), ("ensemble", "thr_ensemble"))
EXPLORATORY_COMPARISONS = (("shallow_asis", "shallow_seedavg"), ("xgb_fitonly", "xgb_asis"),
                           ("xgb_fitonly", "thr_xgb_fitonly"), ("deep_asis", "thr_deep_asis"),
                           ("shallow_asis", "thr_shallow_asis"), ("deep_asis", "shallow_asis"))
SEEDS = (42, 43, 44)   # training/train_neural_retention.py:30; absent seed directories are listed as missing
XGB_REQUIRED = ("probs_val.npy", "probs_test_fitonly.npy", "probs_test_refit.npy", "grid.json", "y_val.npy", "y_test.npy")
XGB_FILES = XGB_REQUIRED + ("result.json",)   # result.json is informational only
SEED_FILES = ("logits_val.npy", "logits_test.npy", "info.json")
EPS = 1e-12


# ----------------------------------------------------------------------------- calibration / mixing
def softmax(logits, temperature: float = 1.0) -> np.ndarray:
    # copy of training/train_neural_retention.py:62-65
    logits = np.asarray(logits, dtype=np.float64) / temperature
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """Bounded validation-NLL minimisation over log T in [log 0.25, log 10].

    Mirror of training/train_neural_retention.py:149-159. The logits keep their dtype (float32 from torch)
    so the temperature matches the shipped one; float64 inputs (log-probabilities) work the same way.
    """
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


def log_probs(probs) -> np.ndarray:
    """Logits for temperature scaling of a model that only exposes probabilities (XGBoost, averages)."""
    return np.log(np.clip(np.asarray(probs, dtype=np.float64), EPS, 1.0))


def average(prob_list) -> np.ndarray:
    return np.mean([np.asarray(p, dtype=np.float64) for p in prob_list], axis=0)


def blend(xgb, neural, w: float) -> np.ndarray:
    return w * np.asarray(xgb, dtype=np.float64) + (1.0 - w) * np.asarray(neural, dtype=np.float64)


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


def apply_class_weights(probs, weights) -> np.ndarray:
    return np.argmax(np.asarray(probs, dtype=np.float64) * np.asarray(weights, dtype=np.float64), axis=1)


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
    """One (resamples, n) index matrix shared by every variant of a dataset, so comparisons are paired."""
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


def paired_macro_f1_delta(y, pred_base, pred_candidate, indices, *, group_ids=None) -> dict:
    """Observed candidate - baseline macro-F1, percentile CI and one-sided bootstrap p (H1: candidate better)."""
    if group_ids is None:
        base, cand = bootstrap_macro_f1(y, pred_base, indices), bootstrap_macro_f1(y, pred_candidate, indices)
    else:
        base = sentence_bootstrap_macro_f1(y, pred_base, group_ids, indices)
        cand = sentence_bootstrap_macro_f1(y, pred_candidate, group_ids, indices)
    deltas = cand - base
    return {"observed": macro_f1(y, pred_candidate) - macro_f1(y, pred_base), "ci95": percentile_ci(deltas),
            "p_one_sided": float((np.sum(deltas <= 0) + 1) / (len(deltas) + 1)), "resamples": int(len(deltas))}


def paired_accuracy_delta(y, pred_base, pred_candidate, *, resamples: int = RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    y = np.asarray(y)
    return paired_comparison((np.asarray(pred_base) == y).astype(float).tolist(),
                             (np.asarray(pred_candidate) == y).astype(float).tolist(), resamples=resamples, seed=seed)


def sentence_level_metrics(y, pred, group_ids, class_names) -> dict:
    """Secondary table: one row per test sentence, label = test-majority class, prediction = majority prediction."""
    labels = oracle_predictions(group_ids, y)
    preds = oracle_predictions(group_ids, pred)
    _, first = np.unique(np.asarray(group_ids), return_index=True)
    return classification_metrics(labels[first], preds[first], class_names=class_names)


# ----------------------------------------------------------------------------- stage-1/2 sources
def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _check_probs(array, n, name) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.shape != (n, 3) or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected finite ({n}, 3) probabilities, got {array.shape}")
    if not np.allclose(array.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(f"{name}: rows do not sum to 1")
    return array


def _check_logits(array, n, name) -> np.ndarray:
    array = np.asarray(array)   # dtype kept (float32) so fit_temperature matches neural_run and the shipped code
    if array.shape != (n, 3) or not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected finite ({n}, 3) float logits, got {array.shape} {array.dtype}")
    return array


def _check_labels(path: Path, name: str, expected) -> None:
    if not np.array_equal(np.load(path / name), expected):
        raise AssertionError(f"{path.name}/{name} differs from the stage-0 labels")


def load_sources(dataset_dir: Path, labels, splits, *, tertile_labels=None) -> tuple[dict, list[str]]:
    """Load whatever stage-1/2 outputs exist; every y_*.npy found must equal the stage-0 labels (else raises)."""
    dataset_dir = Path(dataset_dir)
    y_val, y_test = labels[splits["val"]], labels[splits["test"]]
    sources = {"xgboost": None, "xgboost_tertile": None, "neural": {}, "norm": None, "y_test_sources": []}
    missing = []
    expected_xgb = [("xgboost", y_val, y_test)]
    if tertile_labels is not None:
        expected_xgb.append(("xgboost_tertile", tertile_labels[splits["val"]], tertile_labels[splits["test"]]))
    for folder, ref_val, ref_test in expected_xgb:
        path = dataset_dir / folder
        absent = [name for name in XGB_FILES if not (path / name).exists()]
        missing += [f"{folder}/{name}" for name in absent]
        if any(name in XGB_REQUIRED for name in absent):
            continue
        _check_labels(path, "y_val.npy", ref_val)
        _check_labels(path, "y_test.npy", ref_test)
        sources["y_test_sources"].append(f"{folder}/y_test.npy")
        grid = _read_json(path / "grid.json")
        sources[folder] = {
            "probs_val": _check_probs(np.load(path / "probs_val.npy"), len(ref_val), f"{folder}/probs_val"),
            "probs_test_fitonly": _check_probs(np.load(path / "probs_test_fitonly.npy"), len(ref_test), f"{folder}/probs_test_fitonly"),
            "probs_test_refit": _check_probs(np.load(path / "probs_test_refit.npy"), len(ref_test), f"{folder}/probs_test_refit"),
            "trials": grid.get("trials", []), "chosen": grid.get("chosen", {}), "result": _read_json(path / "result.json")}

    neural = dataset_dir / "neural"
    for name, ref in (("y_val.npy", y_val), ("y_test.npy", y_test)):
        if not (neural / name).exists():
            missing.append(f"neural/{name}")
            continue
        _check_labels(neural, name, ref)
        if name == "y_test.npy":
            sources["y_test_sources"].append("neural/y_test.npy")
    sources["norm"] = _read_json(neural / "norm.json")
    if sources["norm"] is None:
        missing.append("neural/norm.json")
    for short, arch in ARCHITECTURES.items():
        arch_dir = neural / arch
        if not arch_dir.is_dir():
            missing.append(f"neural/{arch}")
            continue
        seed_dirs = {int(path.name.split("_", 1)[1]): path for path in arch_dir.glob("seed_*")
                     if path.is_dir() and path.name.split("_", 1)[1].isdigit()}
        missing += [f"neural/{arch}/seed_{seed}" for seed in SEEDS if seed not in seed_dirs]
        seeds = {}
        for seed in sorted(seed_dirs):
            seed_dir = seed_dirs[seed]
            absent = [name for name in SEED_FILES if not (seed_dir / name).exists()]
            if absent:
                missing += [f"neural/{arch}/{seed_dir.name}/{name}" for name in absent]
                continue
            info = _read_json(seed_dir / "info.json")
            logits_val = _check_logits(np.load(seed_dir / "logits_val.npy"), len(y_val), f"{seed_dir}/logits_val")
            logits_test = _check_logits(np.load(seed_dir / "logits_test.npy"), len(y_test), f"{seed_dir}/logits_test")
            info.setdefault("seed", seed)
            info.setdefault("validation_macro_f1", macro_f1(y_val, logits_val.argmax(axis=1)))
            info.setdefault("validation_nll", nll(y_val, softmax(logits_val)))
            seeds[seed] = {"info": info, "logits_val": logits_val, "logits_test": logits_test}
        result = _read_json(arch_dir / "result.json")
        if result is None:
            missing.append(f"neural/{arch}/result.json")
        if seeds:
            sources["neural"][short] = {"architecture": arch, "seeds": seeds, "result": result}
    return sources, missing


# ----------------------------------------------------------------------------- variants
class Variant:
    def __init__(self, name, *, probs_val=None, probs_test=None, pred_val=None, pred_test=None, tuned=None,
                 based_on=(), exploratory=False, labels="main", val_macro_f1=None, val_nll=None):
        self.name, self.tuned, self.based_on, self.exploratory, self.labels = name, tuned or {}, list(based_on), exploratory, labels
        self.probs_val, self.probs_test = probs_val, probs_test
        self.pred_val = pred_val if pred_val is not None else (probs_val.argmax(axis=1) if probs_val is not None else None)
        self.pred_test = pred_test if pred_test is not None else probs_test.argmax(axis=1)
        self.val_macro_f1, self.val_nll = val_macro_f1, val_nll


def _threshold_variant(base: Variant, y_val) -> Variant:
    weights, info = search_class_weights(base.probs_val, y_val)
    return Variant(f"thr_{base.name}", probs_val=base.probs_val, probs_test=base.probs_test,
                   pred_val=apply_class_weights(base.probs_val, weights),
                   pred_test=apply_class_weights(base.probs_test, weights),
                   tuned={**info, "base": base.name}, based_on=[base.name], exploratory=base.name != "ensemble")


def _selection_key(info: dict) -> tuple[float, float]:
    return info["validation_macro_f1"], -info["validation_nll"]   # train_neural_retention.py:139


def _neural_variants(short: str, source: dict, y_val) -> tuple[Variant, Variant, dict]:
    seeds = source["seeds"]
    chosen = max(sorted(seeds), key=lambda seed: _selection_key(seeds[seed]["info"]))   # first seed wins ties
    per_seed = {seed: fit_temperature(seeds[seed]["logits_val"], y_val) for seed in sorted(seeds)}
    temperature = per_seed[chosen]
    info = seeds[chosen]["info"]
    asis = Variant(f"{short}_asis", probs_val=softmax(seeds[chosen]["logits_val"], temperature),
                   probs_test=softmax(seeds[chosen]["logits_test"], temperature),
                   tuned={"architecture": source["architecture"], "seed": chosen, "best_epoch": info.get("best_epoch"),
                          "temperature": temperature, "seed_validation_macro_f1": info["validation_macro_f1"]})
    val_avg = average([softmax(seeds[seed]["logits_val"], per_seed[seed]) for seed in per_seed])
    test_avg = average([softmax(seeds[seed]["logits_test"], per_seed[seed]) for seed in per_seed])
    ensemble_t = fit_temperature(log_probs(val_avg), y_val)
    seedavg = Variant(f"{short}_seedavg", probs_val=softmax(log_probs(val_avg), ensemble_t),
                      probs_test=softmax(log_probs(test_avg), ensemble_t),
                      tuned={"architecture": source["architecture"], "seeds": sorted(per_seed),
                             "per_seed_temperature": {str(seed): per_seed[seed] for seed in per_seed},
                             "ensemble_temperature": ensemble_t},
                      based_on=[f"{short}_asis"], exploratory=short == "shallow")
    result = source.get("result") or {}
    logged = {seed: seeds[seed]["info"].get("temperature") for seed in per_seed}
    consistency = {
        "selected_seed_matches_result": (result["selected_seed"] == chosen) if "selected_seed" in result else None,
        "temperature_max_abs_diff_vs_info": (max(abs(per_seed[s] - logged[s]) for s in per_seed)
                                             if all(value is not None for value in logged.values()) else None)}
    return asis, seedavg, consistency


def _xgboost_variants(source: dict, y_val) -> tuple[Variant, Variant]:
    chosen = source["chosen"]
    val_f1, val_nll = macro_f1(y_val, source["probs_val"].argmax(axis=1)), nll(y_val, source["probs_val"])
    # the refit model has no out-of-sample validation rows; its val columns repeat the winning fit-only trial
    asis = Variant("xgb_asis", probs_test=source["probs_test_refit"], tuned={**chosen, "refit": "fit+val"},
                   val_macro_f1=val_f1, val_nll=val_nll)
    fitonly = Variant("xgb_fitonly", probs_val=source["probs_val"], probs_test=source["probs_test_fitonly"],
                      tuned={**chosen, "refit": None})
    return asis, fitonly


def build_variants(dataset_dir: Path, *, resamples: int = RESAMPLES) -> dict:
    """Stage 3 for one dataset directory; writes and returns combined.json (raises on a failed hard self-check)."""
    dataset_dir = Path(dataset_dir)
    dataset, _, labels, splits = load_stage0(dataset_dir)
    class_names = dataset["class_names"]
    by_sentence = bool(dataset.get("grouped_by_sentence"))
    y_val, y_test = labels[splits["val"]], labels[splits["test"]]
    group_test = splits["group_ids"][splits["test"]]
    tertile = splits.get("tertile_labels")
    sources, missing = load_sources(dataset_dir, labels, splits, tertile_labels=tertile)

    variants: dict[str, Variant] = {}
    majority = dataset["diagnostics"]["majority"]["class_index"]
    variants["majority"] = Variant("majority", pred_val=np.full_like(y_val, majority), pred_test=np.full_like(y_test, majority),
                                   tuned={"class_index": majority, "class_name": class_names[majority]})
    if by_sentence:
        variants["oracle"] = Variant("oracle", pred_test=oracle_predictions(group_test, y_test), tuned={"uses": "test labels"})
    if sources["xgboost"]:
        asis, fitonly = _xgboost_variants(sources["xgboost"], y_val)
        variants[asis.name], variants[fitonly.name] = asis, fitonly
    consistency = {}
    for short in ARCHITECTURES:
        if short in sources["neural"]:
            asis, seedavg, consistency[short] = _neural_variants(short, sources["neural"][short], y_val)
            variants[asis.name], variants[seedavg.name] = asis, seedavg
    blend_table = None
    checks_blend = {"w1_reproduces_xgb_fitonly": None, "w0_reproduces_deep_seedavg": None}
    if "xgb_fitonly" in variants and "deep_seedavg" in variants:
        xgb, deep = variants["xgb_fitonly"], variants["deep_seedavg"]
        xgb_t = fit_temperature(log_probs(xgb.probs_val), y_val)
        xgb_val_cal, xgb_test_cal = softmax(log_probs(xgb.probs_val), xgb_t), softmax(log_probs(xgb.probs_test), xgb_t)
        w, blend_table = search_blend_weight(xgb_val_cal, deep.probs_val, y_val)
        variants["ensemble"] = Variant("ensemble", probs_val=blend(xgb_val_cal, deep.probs_val, w),
                                       probs_test=blend(xgb_test_cal, deep.probs_test, w),
                                       tuned={"w": w, "xgb_temperature": xgb_t, "blend_table": blend_table,
                                              "members": ["xgb_fitonly (temperature-scaled)", "deep_seedavg"]},
                                       based_on=["xgb_fitonly", "deep_seedavg"])
        only_xgb, only_deep = blend(xgb_test_cal, deep.probs_test, 1.0), blend(xgb_test_cal, deep.probs_test, 0.0)
        checks_blend = {"w1_reproduces_xgb_fitonly": bool(np.allclose(only_xgb, xgb_test_cal)
                                                          and np.array_equal(only_xgb.argmax(axis=1), xgb.pred_test)),
                        "w0_reproduces_deep_seedavg": bool(np.allclose(only_deep, deep.probs_test)
                                                           and np.array_equal(only_deep.argmax(axis=1), deep.pred_test))}
    for base in ("xgb_fitonly", "deep_asis", "shallow_asis", "ensemble"):
        if base in variants:
            variants[f"thr_{base}"] = _threshold_variant(variants[base], y_val)
    if sources["xgboost_tertile"]:
        source = sources["xgboost_tertile"]
        variants["xgb_tertile"] = Variant("xgb_tertile", probs_val=source["probs_val"], probs_test=source["probs_test_refit"],
                                          tuned={**source["chosen"], "refit": "fit+val", "bin_edges": dataset.get("tertile_edges")},
                                          exploratory=True, labels="tertile")

    # ---- metrics and bootstraps (one shared index matrix per dataset, paired across variants)
    row_indices = bootstrap_indices(len(y_test), resamples=resamples)
    n_sentences = len(np.unique(group_test))
    sentence_indices = bootstrap_indices(n_sentences, resamples=resamples) if by_sentence else None
    not_defined = {"oracle": not by_sentence, "xgb_tertile": tertile is None}
    records, preds = {}, {}
    for name in VARIANT_ORDER:
        if name not in variants:
            records[name] = ({"name": name, "status": "n/a", "reason": "not defined for this dataset"} if not_defined.get(name)
                             else {"name": name, "status": "missing", "reason": "stage output absent or incomplete"})
            continue
        v = variants[name]
        truth_val = y_val if v.labels == "main" else tertile[splits["val"]]
        truth_test = y_test if v.labels == "main" else tertile[splits["test"]]
        is_rule = name.startswith("thr_") or name in ("majority", "oracle")
        if v.probs_test is not None and not is_rule:
            test = full_metrics(truth_test, v.probs_test, class_names, include_proxy=dataset["kind"] == "msc")
        else:
            test = {**classification_metrics(truth_test, v.pred_test, class_names=class_names), "nll": None, "ece_10_bins": None}
        val = {"macro_f1": v.val_macro_f1 if v.val_macro_f1 is not None else (macro_f1(truth_val, v.pred_val) if v.pred_val is not None else None),
               "nll": v.val_nll if v.val_nll is not None else (nll(truth_val, v.probs_val) if (v.probs_val is not None and not is_rule) else None),
               "optimistic": True}
        records[name] = {
            "name": name, "status": "ok", "exploratory": v.exploratory, "based_on": v.based_on, "labels": v.labels,
            "tuned": v.tuned, "val": val, "test": test,
            "row_bootstrap": {"macro_f1_ci95": percentile_ci(bootstrap_macro_f1(truth_test, v.pred_test, row_indices))},
            "sentence_bootstrap": ({"macro_f1_ci95": percentile_ci(sentence_bootstrap_macro_f1(truth_test, v.pred_test, group_test, sentence_indices)),
                                    "sentences": int(n_sentences)} if by_sentence else None),
            "sentence_level": (sentence_level_metrics(truth_test, v.pred_test, group_test, class_names)
                               if by_sentence and v.labels == "main" else None)}
        preds[name] = v.pred_test

    comparisons = []
    for primary, pairs in ((True, PRIMARY_COMPARISONS), (False, EXPLORATORY_COMPARISONS)):
        for base, cand in pairs:
            entry = {"baseline": base, "candidate": cand, "primary": primary, "alpha": BONFERRONI_ALPHA if primary else None}
            if base not in preds or cand not in preds:
                comparisons.append({**entry, "status": "missing"})
                continue
            delta = paired_macro_f1_delta(y_test, preds[base], preds[cand], row_indices)
            comparisons.append({**entry, "status": "ok", "macro_f1": delta,
                                "macro_f1_sentence": (paired_macro_f1_delta(y_test, preds[base], preds[cand], sentence_indices, group_ids=group_test)
                                                      if by_sentence else None),
                                "accuracy": paired_accuracy_delta(y_test, preds[base], preds[cand], resamples=resamples),
                                "significant_macro_f1": (delta["p_one_sided"] < BONFERRONI_ALPHA) if primary else None})

    # ---- self-checks (plan section 6); None = not applicable with the outputs present
    models = [n for n in preds if n not in ("majority", "oracle", "xgb_tertile")]
    identity_ok = all(np.array_equal(apply_class_weights(variants[n].probs_test, (1.0, 1.0, 1.0)), variants[n].probs_test.argmax(axis=1))
                      for n in models if variants[n].probs_test is not None)
    oracle = records["oracle"]["test"] if by_sentence else None
    varying = ([n for n in models if any(len(set(preds[n][group_test == g].tolist())) > 1 for g in np.unique(group_test))]
               if by_sentence else [])
    checks = {**checks_blend,
              "all_ones_thresholds_reproduce_argmax": bool(identity_ok) if models else None,
              # mismatching y files raise in load_sources, so reaching here with >= 1 file means identical
              "y_test_identical_across_sources": True if sources["y_test_sources"] else None,
              "y_test_sources": ["data/splits.npz"] + sources["y_test_sources"],
              # a model that predicts one class per normalised sentence (true for every real model: identical text gives
              # identical features) cannot beat the oracle's accuracy; models that vary within a sentence are listed instead
              "models_varying_within_sentence": varying,
              "oracle_accuracy_dominates": (all(oracle["accuracy"] + 1e-12 >= records[n]["test"]["accuracy"]
                                                for n in models if n not in varying)
                                            if oracle and models else None),
              # macro-F1 is not a per-row sum, so exceeding the oracle here is possible in principle: warning only
              "oracle_macro_f1_dominates": (all(oracle["macro_f1"] + 1e-12 >= records[n]["test"]["macro_f1"] for n in models)
                                            if oracle and models else None),
              "neural_consistency": consistency}
    combined = {
        "dataset": dataset["name"], "kind": dataset["kind"], "class_names": class_names, "dataset_dir": str(dataset_dir),
        "resamples": resamples, "bootstrap_seed": BOOTSTRAP_SEED, "stage0": dataset,
        "variant_order": list(VARIANT_ORDER), "variants": records, "comparisons": comparisons,
        "blend_table": blend_table,
        "seeds": {short: [sources["neural"][short]["seeds"][s]["info"] for s in sorted(sources["neural"][short]["seeds"])]
                  for short in sources["neural"]},
        "xgboost_trials": {key: (sources[key]["trials"] if sources[key] else None) for key in ("xgboost", "xgboost_tertile")},
        "sources": {"xgboost": sources["xgboost"]["result"] if sources["xgboost"] else None,
                    "xgboost_tertile": sources["xgboost_tertile"]["result"] if sources["xgboost_tertile"] else None,
                    "neural": {short: sources["neural"][short]["result"] for short in sources["neural"]},
                    "norm": sources["norm"]},
        "missing": missing, "self_checks": checks,
    }
    (dataset_dir / "combined.json").write_text(json.dumps(combined, indent=2) + "\n")
    hard = ("w1_reproduces_xgb_fitonly", "w0_reproduces_deep_seedavg", "all_ones_thresholds_reproduce_argmax",
            "oracle_accuracy_dominates")
    failed = [key for key in hard if checks[key] is False]
    if failed:
        raise AssertionError(f"self-checks failed: {failed}")
    return combined


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    args = parser.parse_args(argv)
    combined = build_variants(args.dataset_dir, resamples=args.resamples)
    summary = {name: {"status": r["status"], **({"macro_f1": r["test"]["macro_f1"], "accuracy": r["test"]["accuracy"]} if r["status"] == "ok" else {})}
               for name, r in combined["variants"].items()}
    print(json.dumps({"dataset": combined["dataset"], "variants": summary, "missing": combined["missing"],
                      "self_checks": combined["self_checks"]}, indent=2))


if __name__ == "__main__":
    main()
