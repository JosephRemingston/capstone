"""Stage 2: the ensemble's deep-head seeds. torch (+ onnxruntime for the export check); never imports xgboost.

    python -m training.ensemble.neural_stage --out artifacts/ensemble_training

Trains the frozen-BGE deep head for seeds 42/43/44 with early stopping on validation, saves each seed's
weights and validation/test logits, and exports it to ONNX (inputs ``embedding`` and ``numeric``, output
``logits``), checking onnxruntime against torch.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss

from main.ml import TEXT_FEATURES
from training.ensemble import datasets, features
from training.ensemble.heads import ARCHITECTURE, create_head
from training.ensemble.metrics import macro_f1

SEEDS = (42, 43, 44)
MAX_EPOCHS = 100
PATIENCE = 12
BATCH_SIZE = 32
ONNX_OPSET = 17
ONNX_TOLERANCE = 1e-4


def normalisation(x_numeric, fit_idx) -> tuple[np.ndarray, np.ndarray]:
    """Fit-split mean / std with near-zero std set to 1."""
    mean = x_numeric[fit_idx].mean(axis=0)
    scale = x_numeric[fit_idx].std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def logits_of(model, embeddings: np.ndarray, numeric_scaled: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return model(torch.from_numpy(embeddings), torch.from_numpy(numeric_scaled)).numpy()


def _probabilities(model, embeddings, numeric_scaled):
    # float32 softmax, so early stopping sees the same numbers as the original trainer
    logits = logits_of(model, embeddings, numeric_scaled)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def train_head(seed, x_embed, x_numeric, labels, fit_idx, val_idx, mean, scale, *,
               max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE):
    """AdamW, sqrt-balanced class weights, early stopping on (val macro-F1, -val NLL); kwargs let tests shorten it."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = create_head()
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


def export_onnx(model, path: Path, check_embeddings: np.ndarray, check_numeric: np.ndarray) -> float:
    """Export with the serving interface and return the largest |onnxruntime - torch| logit difference."""
    import onnxruntime as ort
    model.eval()
    torch.onnx.export(
        model, (torch.zeros(1, 384), torch.zeros(1, len(TEXT_FEATURES))), path,
        input_names=["embedding", "numeric"], output_names=["logits"],
        dynamic_axes={"embedding": {0: "batch"}, "numeric": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=ONNX_OPSET, dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    exported = session.run(["logits"], {"embedding": check_embeddings, "numeric": check_numeric})[0]
    reference = logits_of(model, check_embeddings, check_numeric)
    if not np.array_equal(exported.argmax(axis=1), reference.argmax(axis=1)):
        raise AssertionError(f"{path.name}: ONNX export changes predictions")
    difference = float(np.abs(exported - reference).max())
    if difference > ONNX_TOLERANCE:
        raise AssertionError(f"{path.name}: ONNX logits differ from torch by {difference}")
    return difference


def run(out_dir: Path, *, seeds=SEEDS, threads: int = 4, max_epochs: int = MAX_EPOCHS) -> dict:
    started = time.perf_counter()
    torch.set_num_threads(min(threads, torch.get_num_threads()))
    out_dir = Path(out_dir)
    dataset, rows, labels, splits = datasets.load_stage0(out_dir)
    fit_idx, val_idx, test_idx = splits["fit"], splits["val"], splits["test"]
    labels = labels.astype(np.int64)
    redacted, x_numeric = features.neural_features([row["text"] for row in rows])
    x_embed = features.cached_embeddings(redacted, out_dir / "embeddings", threads=threads)
    mean, scale = normalisation(x_numeric, fit_idx)
    scaled = ((x_numeric - mean) / scale).astype(np.float32)
    out = out_dir / "neural"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "y_val.npy", labels[val_idx])
    np.save(out / "y_test.npy", labels[test_idx])
    (out / "norm.json").write_text(json.dumps({"numeric_mean": mean.tolist(), "numeric_scale": scale.tolist()}) + "\n")
    members = []
    for seed in seeds:
        model, info = train_head(seed, x_embed, x_numeric, labels, fit_idx, val_idx, mean, scale, max_epochs=max_epochs)
        seed_dir = out / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        np.save(seed_dir / "logits_val.npy", logits_of(model, x_embed[val_idx], scaled[val_idx]))
        np.save(seed_dir / "logits_test.npy", logits_of(model, x_embed[test_idx], scaled[test_idx]))
        torch.save(model.state_dict(), seed_dir / "model.pt")
        check = np.concatenate([val_idx, test_idx])
        info["onnx_max_abs_logit_diff"] = export_onnx(model, seed_dir / "model.onnx", x_embed[check], scaled[check])
        info["onnx_sha256"] = hashlib.sha256((seed_dir / "model.onnx").read_bytes()).hexdigest()
        info["weights_sha256"] = hashlib.sha256((seed_dir / "model.pt").read_bytes()).hexdigest()
        (seed_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n")
        members.append(info)
    result = {"status": "ok", "architecture": ARCHITECTURE, "seconds": round(time.perf_counter() - started, 1),
              "torch_version": torch.__version__, "encoder_id": features.encoder_id(),
              "fastembed_version": version("fastembed"), "threads": torch.get_num_threads(), "onnx_opset": ONNX_OPSET,
              "fit_size": len(fit_idx), "validation_size": len(val_idx), "test_size": len(test_idx), "seeds": members}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=datasets.DEFAULT_OUT)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args(argv)
    result = run(args.out, seeds=tuple(int(s) for s in args.seeds.split(",") if s), threads=args.threads,
                 max_epochs=args.max_epochs)
    print(json.dumps({"seconds": result["seconds"],
                      "seeds": [{k: s[k] for k in ("seed", "best_epoch", "validation_macro_f1", "onnx_max_abs_logit_diff")}
                                for s in result["seeds"]]}))


if __name__ == "__main__":
    main()
