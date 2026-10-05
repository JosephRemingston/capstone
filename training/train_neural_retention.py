"""Train the frozen-BGE, two-branch neural retention proxy."""

from __future__ import annotations

import copy
import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.model_selection import train_test_split

from main.domain.classification import MemoryClassifier
from main.domain.models import MemoryInput, MemoryRecord
from main.domain.privacy import redact_sensitive
from main.ml import MLFeatureExtractor, TEXT_FEATURES
from main.neural import DEFAULT_NEURAL_DIR, SCORE_VALUES, NeuralImportanceScorer, create_head
from main.retrieval.embeddings import FastEmbedder
from training.train_conversational_retention import (CLASSES, FILES, REPORT_DIR,
                                                      load_rows, metrics, target)


SEEDS = (42, 43, 44)
SPLIT_SEED = 42
MAX_EPOCHS = 100
PATIENCE = 12
BATCH_SIZE = 32


def prepare_neural(rows: list[dict], embedder: FastEmbedder):
    classifier = MemoryClassifier(legacy=True)
    extractor = MLFeatureExtractor()
    texts, numeric, labels = [], [], []
    for row in rows:
        label = target(row)
        if label is None or not row["text"].strip():
            continue
        text, _ = redact_sensitive(row["text"])
        incoming = MemoryInput(content=text, user_id="dataset", session_id="dataset")
        record = MemoryRecord.from_input(incoming, classifier.classify(incoming))
        features = extractor.extract(incoming, record)
        texts.append(text)
        numeric.append([features[name] for name in TEXT_FEATURES])
        labels.append(label)
    embeddings = np.asarray(embedder.passages(texts), dtype=np.float32)
    if embeddings.shape != (len(labels), 384) or not np.isfinite(embeddings).all():
        raise ValueError("Invalid BGE embeddings")
    return texts, embeddings, np.asarray(numeric, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def probabilities(model, embeddings, numeric, temperature=1.0):
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(embeddings), torch.from_numpy(numeric)).numpy()
    logits = logits / temperature
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def ece(labels, probs, bins=10):
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    return float(sum((mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
                     for i in range(bins)
                     if (mask := (confidence >= edges[i]) &
                         (confidence <= edges[i + 1] if i == bins - 1
                          else confidence < edges[i + 1])).any()))


def train_one(seed, x_embed, x_numeric, labels, fit_idx, val_idx, mean, scale,
              architecture="frozen-bge-two-branch-v1"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = create_head(architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    counts = Counter(labels[fit_idx].tolist())
    weights = np.asarray([math.sqrt(len(fit_idx) / (3 * counts[i])) for i in range(3)], dtype=np.float32)
    weights /= weights.mean()
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(weights))
    fit_embeddings = torch.from_numpy(x_embed[fit_idx])
    fit_numeric = torch.from_numpy(((x_numeric[fit_idx] - mean) / scale).astype(np.float32))
    fit_labels = torch.from_numpy(labels[fit_idx])
    val_numeric = ((x_numeric[val_idx] - mean) / scale).astype(np.float32)
    best = None
    patience = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(fit_idx), generator=torch.Generator().manual_seed(seed * 1000 + epoch))
        for indexes in order.split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            logits = model(fit_embeddings[indexes], fit_numeric[indexes])
            loss = loss_fn(logits, fit_labels[indexes])
            loss.backward()
            optimizer.step()
        val_probs = probabilities(model, x_embed[val_idx], val_numeric)
        val_f1 = metrics(labels[val_idx], val_probs.argmax(axis=1))["macro_f1"]
        val_loss = float(log_loss(labels[val_idx], val_probs, labels=[0, 1, 2]))
        key = (val_f1, -val_loss)
        if best is None or key > best[0]:
            best = (key, epoch, copy.deepcopy(model.state_dict()))
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    model.load_state_dict(best[2])
    return model, {"seed": seed, "best_epoch": best[1],
                   "validation_macro_f1": best[0][0], "validation_nll": -best[0][1]}


def run(*, architecture="frozen-bge-deep-head-v1", artifact_dir=DEFAULT_NEURAL_DIR,
        report_name="neural_retention.json"):
    artifact_dir = Path(artifact_dir)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    train_rows, test_rows = load_rows("train"), load_rows("test")
    embedder = FastEmbedder()
    _, x_embed, x_numeric, labels = prepare_neural(train_rows, embedder)
    test_texts, test_embed, test_numeric, test_labels = prepare_neural(test_rows, embedder)
    fit_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=0.2,
                                        stratify=labels, random_state=SPLIT_SEED)
    mean = x_numeric[fit_idx].mean(axis=0)
    scale = x_numeric[fit_idx].std(axis=0)
    scale[scale < 1e-8] = 1.0
    runs = []
    best = None
    for seed in SEEDS:
        model, summary = train_one(seed, x_embed, x_numeric, labels,
                                   fit_idx, val_idx, mean, scale, architecture)
        runs.append(summary)
        key = (summary["validation_macro_f1"], -summary["validation_nll"])
        if best is None or key > best[0]:
            best = (key, model, summary)
    chosen_model = best[1]
    val_numeric = ((x_numeric[val_idx] - mean) / scale).astype(np.float32)
    chosen_model.eval()
    with torch.inference_mode():
        val_logits = chosen_model(torch.from_numpy(x_embed[val_idx]),
                                  torch.from_numpy(val_numeric)).numpy()

    def nll_at(log_temperature):
        temperature = math.exp(log_temperature)
        shifted = val_logits / temperature
        shifted -= shifted.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        probs = exponent / exponent.sum(axis=1, keepdims=True)
        return log_loss(labels[val_idx], probs, labels=[0, 1, 2])

    calibrated = minimize_scalar(nll_at, bounds=(math.log(0.25), math.log(10.0)),
                                 method="bounded")
    temperature = float(math.exp(calibrated.x))
    test_scaled = ((test_numeric - mean) / scale).astype(np.float32)
    test_probs = probabilities(chosen_model, test_embed, test_scaled, temperature)
    test_result = metrics(test_labels, test_probs.argmax(axis=1))
    test_result.update({
        "proxy_mae": float(mean_absolute_error(SCORE_VALUES[test_labels],
                                               test_probs @ SCORE_VALUES)),
        "nll": float(log_loss(test_labels, test_probs, labels=[0, 1, 2])),
        "ece_10_bins": ece(test_labels, test_probs),
    })

    # The frozen XGBoost result is already recorded on this published test split.
    # Loading XGBoost and PyTorch native libraries in one macOS process can crash.
    frozen_report = json.loads((REPORT_DIR / "conversational_retention.json").read_text())
    expected_counts = {CLASSES[i]: int((test_labels == i).sum()) for i in range(3)}
    if (frozen_report["test_class_counts"] != expected_counts or
            frozen_report["dataset_size"]["test_used"] != len(test_labels)):
        raise ValueError("Frozen XGBoost baseline uses a different test split")
    xgb_result = frozen_report["test"]
    majority = int(np.argmax(np.bincount(labels)))
    majority_result = metrics(test_labels, np.full_like(test_labels, majority))

    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / "model.pt"
    torch.save(chosen_model.state_dict(), weights_path)
    model_path = artifact_dir / "model.onnx"
    torch.onnx.export(
        chosen_model, (torch.zeros(1, 384), torch.zeros(1, len(TEXT_FEATURES))),
        model_path, input_names=["embedding", "numeric"], output_names=["logits"],
        dynamic_axes={"embedding": {0: "batch"}, "numeric": {0: "batch"},
                      "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    metadata = {
        "architecture": architecture,
        "target": "conversational_retention_proxy_v1",
        "class_order": list(CLASSES), "score_values": SCORE_VALUES.tolist(),
        "feature_names": list(TEXT_FEATURES), "numeric_mean": mean.tolist(),
        "numeric_scale": scale.tolist(), "encoder_id": embedder.model_id,
        "temperature": temperature,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "training_weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "source_file_sha256": {key: value[0] for key, value in FILES.items()},
        "selected_seed": best[2]["seed"], "selected_epoch": best[2]["best_epoch"],
        "torch_version": torch.__version__,
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    # Include the encoder in production latency, after warming model and cache.
    scorer = NeuralImportanceScorer(artifact_dir, embedder=embedder)
    classifier = MemoryClassifier(legacy=True)
    extractor = MLFeatureExtractor()
    for text in test_texts[:2]:
        incoming = MemoryInput(content=text, user_id="latency", session_id="latency")
        record = MemoryRecord.from_input(incoming, classifier.classify(incoming))
        scorer.score_text(text, extractor.extract(incoming, record))
    latencies = []
    for text in test_texts[:50]:
        incoming = MemoryInput(content=text, user_id="latency", session_id="latency")
        record = MemoryRecord.from_input(incoming, classifier.classify(incoming))
        features = extractor.extract(incoming, record)
        start = time.perf_counter()
        scorer.score_text(text, features)
        latencies.append((time.perf_counter() - start) * 1000)

    report = {
        "dataset": "https://huggingface.co/datasets/adugeen/personal-facts-msc",
        "architecture": architecture,
        "target": "human-labeled validity/duration; score is a constructed retention proxy",
        "train_used": len(labels), "fit_size": len(fit_idx), "validation_size": len(val_idx),
        "test_size": len(test_labels), "seeds": runs,
        "selected_seed": best[2]["seed"], "temperature": temperature,
        "validation_nll_before_calibration": float(nll_at(0.0)),
        "validation_nll_after_calibration": float(nll_at(math.log(temperature))),
        "neural_test": test_result, "xgboost_test": xgb_result,
        "majority_test": majority_result,
        "latency_ms": {"p50": float(np.percentile(latencies, 50)),
                       "p95": float(np.percentile(latencies, 95)),
                       "sample_count": len(latencies),
                       "scope": "warm local encoder plus neural scorer, single fact"},
        "caveat": "Exploratory comparison: published test split was previously inspected for XGBoost; no human numeric importance labels.",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / report_name).write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("frozen-bge-two-branch-v1",
                                                  "frozen-bge-deep-head-v1"),
                        default="frozen-bge-deep-head-v1")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_NEURAL_DIR)
    parser.add_argument("--report-name", default="neural_retention.json")
    args = parser.parse_args()
    result = run(architecture=args.architecture, artifact_dir=args.artifact_dir,
                 report_name=args.report_name)
    print(json.dumps({"selected_seed": result["selected_seed"],
                      "neural_test": result["neural_test"],
                      "xgboost_test": result["xgboost_test"],
                      "majority_test": result["majority_test"],
                      "latency_ms": result["latency_ms"]}, indent=2))
