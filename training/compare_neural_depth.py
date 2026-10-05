"""Compare the deployed and deeper neural heads using training data only."""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from main.neural import create_head
from main.retrieval.embeddings import FastEmbedder
from training.train_conversational_retention import REPORT_DIR, load_rows
from training.train_neural_retention import SEEDS, SPLIT_SEED, prepare_neural, train_one


ARCHITECTURES = ("frozen-bge-two-branch-v1", "frozen-bge-deep-head-v1")


def run():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    _, embedding, numeric, labels = prepare_neural(load_rows("train"), FastEmbedder())
    fit_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=0.2,
                                        stratify=labels, random_state=SPLIT_SEED)
    mean = numeric[fit_idx].mean(axis=0)
    scale = numeric[fit_idx].std(axis=0)
    scale[scale < 1e-8] = 1.0
    results = {}
    for architecture in ARCHITECTURES:
        trials = []
        for seed in SEEDS:
            _, summary = train_one(seed, embedding, numeric, labels,
                                   fit_idx, val_idx, mean, scale, architecture)
            trials.append(summary)
        best = max(trials, key=lambda item: (item["validation_macro_f1"],
                                              -item["validation_nll"]))
        head = create_head(architecture)
        results[architecture] = {
            "parameter_count": sum(parameter.numel() for parameter in head.parameters()),
            "trials": trials, "selected": best,
            "mean_validation_macro_f1": float(np.mean(
                [trial["validation_macro_f1"] for trial in trials])),
        }
    report = {
        "dataset": "Personal Facts (MSC) published train split only",
        "fit_size": len(fit_idx), "validation_size": len(val_idx),
        "comparison_rule": "Select by validation macro F1, then lower validation NLL; published test split untouched",
        "results": results,
        "preferred_architecture": max(ARCHITECTURES, key=lambda name: (
            results[name]["selected"]["validation_macro_f1"],
            -results[name]["selected"]["validation_nll"])),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "neural_depth_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
