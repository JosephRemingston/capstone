"""Tier-1 gate: score the MSC test rows with the SHIPPED artifacts through this harness's data path.

The confusion matrices must equal the shipped reports exactly; a mismatch means the harness's loader,
row order, redaction, features or embeddings differ from the shipped training code.

    python -m training.experiments.replay_shipped --model xgboost
    python -m training.experiments.replay_shipped --model neural --dir artifacts/importance_neural
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.experiments import datasets, features
from training.experiments.metrics import full_metrics

DEFAULT_DATASET_DIR = datasets.ROOT / "artifacts" / "experiments" / "retention_datasets" / "msc"
EXPECTED = {   # test confusion matrices from evaluation/reports/{conversational,neural,neural_shallow}_retention.json
    "importance": [[43, 17, 25], [11, 68, 13], [49, 56, 274]],
    "importance_neural": [[36, 18, 31], [5, 68, 19], [20, 29, 330]],
    "importance_neural_shallow": [[27, 26, 32], [2, 71, 19], [12, 31, 336]],
}


def xgboost_probs(model_dir: Path, texts: list[str]) -> np.ndarray:
    import xgboost as xgb   # this process loads xgboost only
    booster = xgb.Booster()
    booster.load_model(str(model_dir / "model.ubj"))
    return np.asarray(booster.inplace_predict(features.xgboost_features(texts)), dtype=np.float64)


def neural_probs(model_dir: Path, texts: list[str], cache_dir: Path) -> np.ndarray:
    """Batched form of main/neural.py NeuralImportanceScorer.predict_proba (onnxruntime; no torch)."""
    from main.neural import NeuralImportanceScorer
    scorer = NeuralImportanceScorer(model_dir)   # validates checksum, schema, normalisation and temperature
    redacted, numeric = features.neural_features(texts)
    embeddings = features.cached_embeddings(redacted, cache_dir)
    logits = scorer.session.run(["logits"], {"embedding": embeddings,
                                             "numeric": ((numeric - scorer.mean) / scorer.scale).astype(np.float32)})[0]
    logits = logits / scorer.temperature
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponent / exponent.sum(axis=1, keepdims=True)


def run(model: str, model_dir: Path, dataset_dir: Path = DEFAULT_DATASET_DIR) -> dict:
    dataset, rows, labels, splits = datasets.load_stage0(dataset_dir)
    if dataset["kind"] != "msc":
        raise SystemExit("Replay applies to the MSC dataset only")
    test = splits["test"]
    texts = [rows[i]["text"] for i in test]
    probs = (xgboost_probs(model_dir, texts) if model == "xgboost"
             else neural_probs(model_dir, texts, Path(dataset_dir) / "embeddings"))
    result = full_metrics(labels[test], probs, dataset["class_names"], include_proxy=True)
    expected = EXPECTED.get(model_dir.name)
    report = {"model": model, "artifact_dir": str(model_dir), "test_rows": len(test), "metrics": result,
              "expected_confusion_matrix": expected,
              "exact_match": expected is not None and result["confusion_matrix"] == expected}
    out = Path(dataset_dir) / "replay"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{model_dir.name}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("xgboost", "neural"), required=True)
    parser.add_argument("--dir", type=Path, default=None, help="artifact directory (default: the shipped one)")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    args = parser.parse_args(argv)
    model_dir = args.dir or datasets.ROOT / "artifacts" / ("importance" if args.model == "xgboost" else "importance_neural")
    report = run(args.model, model_dir.resolve(), args.dataset_dir)
    print(json.dumps({"artifact": model_dir.name, "macro_f1": report["metrics"]["macro_f1"],
                      "confusion_matrix": report["metrics"]["confusion_matrix"], "exact_match": report["exact_match"]}))
    if not report["exact_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
