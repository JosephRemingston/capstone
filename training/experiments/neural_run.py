"""Stage 2: the shipped frozen-BGE retention heads on one dataset. torch only; never imports xgboost or training.train_*.

    python -m training.experiments.neural_run --dataset-dir artifacts/experiments/retention_datasets/msc
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss

from main.neural import create_head
from training.experiments import datasets, features
from training.experiments.combine import ARCHITECTURES, fit_temperature, softmax
from training.experiments.metrics import full_metrics, macro_f1

SEEDS = (42, 43, 44)   # training/train_neural_retention.py:30-34
MAX_EPOCHS = 100
PATIENCE = 12
BATCH_SIZE = 32


def normalisation(x_numeric, fit_idx) -> tuple[np.ndarray, np.ndarray]:
    """Fit-split mean / std with near-zero std set to 1, training/train_neural_retention.py:130-132."""
    mean = x_numeric[fit_idx].mean(axis=0)
    scale = x_numeric[fit_idx].std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def logits_of(model, embeddings: np.ndarray, numeric_scaled: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return model(torch.from_numpy(embeddings), torch.from_numpy(numeric_scaled)).numpy()


def _probabilities(model, embeddings, numeric_scaled):
    # float32 softmax exactly as training/train_neural_retention.py:58-65, so early stopping matches the shipped trainer
    logits = logits_of(model, embeddings, numeric_scaled)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def train_head(seed, x_embed, x_numeric, labels, fit_idx, val_idx, mean, scale, architecture, *,
               max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE):
    """Copy of training/train_neural_retention.py:79-117; kwargs exist only so tests can shorten training."""
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
    waited = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(fit_idx), generator=torch.Generator().manual_seed(seed * 1000 + epoch))
        for indexes in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(fit_embeddings[indexes], fit_numeric[indexes]), fit_labels[indexes])
            loss.backward()
            optimizer.step()
        val_probs = _probabilities(model, x_embed[val_idx], val_numeric)
        val_f1 = macro_f1(labels[val_idx], val_probs.argmax(axis=1))
        val_loss = float(log_loss(labels[val_idx], val_probs, labels=[0, 1, 2]))
        key = (val_f1, -val_loss)
        if best is None or key > best[0]:
            best = (key, epoch, copy.deepcopy(model.state_dict()))
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best[2])
    return model, {"seed": seed, "best_epoch": best[1],
                   "validation_macro_f1": best[0][0], "validation_nll": -best[0][1]}


def run(dataset_dir: Path, *, architectures=tuple(ARCHITECTURES.values()), seeds=SEEDS, threads: int = 4,
        max_epochs: int = MAX_EPOCHS) -> dict:
    started = time.perf_counter()
    torch.set_num_threads(min(threads, torch.get_num_threads()))
    dataset_dir = Path(dataset_dir)
    dataset, rows, labels, splits = datasets.load_stage0(dataset_dir)
    class_names, proxy = dataset["class_names"], dataset["kind"] == "msc"
    fit_idx, val_idx, test_idx = splits["fit"], splits["val"], splits["test"]
    labels = labels.astype(np.int64)
    redacted, x_numeric = features.neural_features([row["text"] for row in rows])
    x_embed = features.cached_embeddings(redacted, dataset_dir / "embeddings", threads=threads)
    mean, scale = normalisation(x_numeric, fit_idx)
    scaled = ((x_numeric - mean) / scale).astype(np.float32)
    out = dataset_dir / "neural"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "y_val.npy", labels[val_idx])
    np.save(out / "y_test.npy", labels[test_idx])
    (out / "norm.json").write_text(json.dumps({"numeric_mean": mean.tolist(), "numeric_scale": scale.tolist()}) + "\n")
    results = {}
    for architecture in architectures:
        arch_started = time.perf_counter()
        runs, best = [], None
        for seed in seeds:
            model, info = train_head(seed, x_embed, x_numeric, labels, fit_idx, val_idx, mean, scale, architecture,
                                     max_epochs=max_epochs)
            val_logits = logits_of(model, x_embed[val_idx], scaled[val_idx])
            test_logits = logits_of(model, x_embed[test_idx], scaled[test_idx])
            info["temperature"] = fit_temperature(val_logits, labels[val_idx])
            seed_dir = out / architecture / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            np.save(seed_dir / "logits_val.npy", val_logits)
            np.save(seed_dir / "logits_test.npy", test_logits)
            torch.save(model.state_dict(), seed_dir / "model.pt")
            (seed_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n")
            runs.append(info)
            key = (info["validation_macro_f1"], -info["validation_nll"])
            if best is None or key > best[0]:
                best = (key, info, val_logits, test_logits)
        _, chosen, val_logits, test_logits = best
        temperature = chosen["temperature"]
        result = {"status": "ok", "dataset": dataset["name"], "architecture": architecture, "class_names": class_names,
                  "seconds": round(time.perf_counter() - arch_started, 1), "torch_version": torch.__version__,
                  "encoder_id": features.encoder_id(), "fastembed_version": version("fastembed"),
                  "threads": torch.get_num_threads(), "fit_size": len(fit_idx), "validation_size": len(val_idx),
                  "test_size": len(test_idx), "seeds": runs, "selected_seed": chosen["seed"],
                  "selected_epoch": chosen["best_epoch"], "temperature": temperature,
                  "validation_nll_before_calibration": chosen["validation_nll"],
                  "validation_nll_after_calibration": float(log_loss(labels[val_idx], softmax(val_logits, temperature),
                                                                     labels=[0, 1, 2])),
                  "test": full_metrics(labels[test_idx], softmax(test_logits, temperature), class_names,
                                       include_proxy=proxy)}
        (out / architecture / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        results[architecture] = result
    return {"dataset": dataset["name"], "seconds": round(time.perf_counter() - started, 1), "results": results}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--architectures", default=",".join(ARCHITECTURES.values()))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args(argv)
    summary = run(args.dataset_dir, architectures=tuple(a for a in args.architectures.split(",") if a),
                  seeds=tuple(int(s) for s in args.seeds.split(",") if s), threads=args.threads,
                  max_epochs=args.max_epochs)
    print(json.dumps({"dataset": summary["dataset"], "seconds": summary["seconds"],
                      "results": {arch: {"selected_seed": r["selected_seed"], "selected_epoch": r["selected_epoch"],
                                         "temperature": round(r["temperature"], 4),
                                         "test_macro_f1": r["test"]["macro_f1"],
                                         "seed_epochs": [s["best_epoch"] for s in r["seeds"]],
                                         "seed_val_f1": [round(s["validation_macro_f1"], 4) for s in r["seeds"]]}
                                  for arch, r in summary["results"].items()}}))


if __name__ == "__main__":
    main()
