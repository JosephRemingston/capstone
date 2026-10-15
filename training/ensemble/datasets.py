"""Stage 0: the csv5k importance dataset, its low/mid/high labels, the sentence-grouped split and diagnostics.

    python -m training.ensemble.datasets --out artifacts/ensemble_training

Writes ``<out>/data/{dataset.json, rows.json, splits.npz}``. The row order of ``rows.json`` is the index
for every per-row array written by later stages.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from training.ensemble.metrics import classification_metrics, majority_class

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "experiments"             # git-ignored copy of the source CSV
DEFAULT_CSV = DATA_DIR / "csv5k.csv"
DEFAULT_OUT = ROOT / "artifacts" / "ensemble_training"
DATASET_NAME = "csv5k"
CLASS_NAMES = ("low", "mid", "high")
EXAMPLE_SUFFIX = re.compile(r"\s*\(example \d+\)")   # NOT end-anchored: suffix sits before the final period
BIN_EDGES = (0.4, 0.7)                               # low <0.4, mid [0.4,0.7), high >=0.7
PIN_FRACTION = 0.05                                  # groups holding >=5% of rows go to fit (label-blind)
SEED = 42
SPLIT_NAMES = ("fit", "val", "test")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def materialize_csv(source: Path | None) -> tuple[Path, str]:
    """The training CSV inside data/experiments/; a given source (e.g. ~/Downloads/...) is copied there first."""
    if source is None:
        if not DEFAULT_CSV.is_file():
            raise FileNotFoundError(f"{DEFAULT_CSV} not found; pass --csv PATH to cognimem_memory_importance_5000.csv")
        return DEFAULT_CSV, sha256_file(DEFAULT_CSV)
    source = Path(source).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"source CSV not found at {source}; pass --csv PATH")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(source)
    if source.resolve() != DEFAULT_CSV.resolve() and (not DEFAULT_CSV.exists() or sha256_file(DEFAULT_CSV) != digest):
        shutil.copyfile(source, DEFAULT_CSV)
    return DEFAULT_CSV, digest


def normalize_sentence(text: str) -> str:
    """Strip the ' (example N)' generation artefact; used for BOTH grouping and model input."""
    return re.sub(r"\s+", " ", EXAMPLE_SUFFIX.sub("", text)).strip()


def load_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "memory" not in reader.fieldnames or "importance_score" not in reader.fieldnames:
            raise ValueError(f"{path}: expected columns 'memory' and 'importance_score'")
        rows = []
        for raw in reader:
            text = normalize_sentence(raw["memory"])
            if not text:
                raise ValueError(f"{path}: empty memory text")
            rows.append({"text": text, "score": float(raw["importance_score"]), "group": text})
    return rows


def bin_scores(scores, edges=BIN_EDGES) -> np.ndarray:
    return np.searchsorted(np.asarray(edges, dtype=float), np.asarray(scores, dtype=float), side="right").astype(np.int64)


def _check_partition(splits: dict, groups: np.ndarray) -> None:
    n = len(groups)
    joined = np.concatenate([splits[name] for name in SPLIT_NAMES])
    assert len(joined) == n and np.array_equal(np.sort(joined), np.arange(n)), "split is not a partition"
    for left in SPLIT_NAMES:
        for right in SPLIT_NAMES:
            if left < right:
                assert not set(groups[splits[left]]) & set(groups[splits[right]]), f"group overlap {left}/{right}"


def group_codes(groups) -> np.ndarray:
    """Integer group ids in first-appearance (row) order.

    StratifiedGroupKFold's assignment depends on how groups are numbered; row-order codes give the
    published 3,200 / 792 / 1,008 partition. np.unique's sorted-text codes would give a different one.
    """
    _, first, inverse = np.unique(np.asarray(groups), return_index=True, return_inverse=True)
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(first))
    return rank[inverse.ravel()]


def grouped_split(groups, labels, *, seed: int = SEED, pin_fraction: float = PIN_FRACTION) -> dict:
    """Sentence-grouped fit/val/test with the pin rule; returns index arrays (ascending)."""
    groups, labels = np.asarray(groups), np.asarray(labels)
    n = len(groups)
    inverse = group_codes(groups)
    counts = np.bincount(inverse)
    pinned = counts[inverse] >= pin_fraction * n
    pool = np.flatnonzero(~pinned)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    _, test_local = next(folds.split(pool.reshape(-1, 1), labels[pool], inverse[pool]))
    test = pool[test_local]
    remaining = np.setdiff1d(pool, test)
    _, val_local = next(folds.split(remaining.reshape(-1, 1), labels[remaining], inverse[remaining]))
    val = remaining[val_local]
    fit = np.setdiff1d(np.arange(n), np.concatenate([test, val]))
    splits = {"fit": fit, "val": val, "test": np.sort(test)}
    _check_partition(splits, groups)
    assert np.isin(np.flatnonzero(pinned), fit).all()
    return splits


def oracle_predictions(groups, labels) -> np.ndarray:
    """Majority label among the given rows of each group (ties -> lowest index), per row."""
    groups, labels = np.asarray(groups), np.asarray(labels)
    _, inverse = np.unique(groups, return_inverse=True)
    counts = np.zeros((inverse.max() + 1, 3), dtype=np.int64)
    np.add.at(counts, (inverse, labels), 1)
    return counts.argmax(axis=1)[inverse]


def class_counts(labels) -> dict:
    return {name: int((np.asarray(labels) == i).sum()) for i, name in enumerate(CLASS_NAMES)}


def diagnostics(rows: list[dict], labels, splits: dict) -> dict:
    labels = np.asarray(labels)
    groups = np.asarray([row["group"] for row in rows])
    scores = np.asarray([row["score"] for row in rows], dtype=float)
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    multi = np.flatnonzero(counts > 1)
    disagree = sum(1 for g in multi if len(set(labels[inverse == g].tolist())) > 1)
    score_std = [float(np.std(scores[inverse == g])) for g in multi]
    largest = int(counts.argmax())
    majority = majority_class(labels[np.concatenate([splits["fit"], splits["val"]])])
    test_labels = labels[splits["test"]]
    words = [len(row["text"].split()) for row in rows]
    return {
        "rows": len(rows), "unique_sentences": int(len(unique)),
        "singleton_groups": int((counts == 1).sum()), "multi_row_groups": int(len(multi)),
        "largest_group": {"text": str(unique[largest]), "rows": int(counts[largest])},
        "disagreeing_multi_groups": int(disagree),
        "disagreeing_multi_group_share": float(disagree / len(multi)) if len(multi) else None,
        "mean_within_group_score_std": float(np.mean(score_std)) if score_std else None,
        "words_per_text": {"mean": float(np.mean(words)), "max": int(np.max(words))},
        "score_range": [float(scores.min()), float(scores.max())], "score_mean": float(scores.mean()),
        "class_counts": {"all": class_counts(labels),
                         **{name: class_counts(labels[splits[name]]) for name in SPLIT_NAMES}},
        "split_sizes": {name: int(len(splits[name])) for name in SPLIT_NAMES},
        "split_sentences": {name: int(len(set(groups[splits[name]]))) for name in SPLIT_NAMES},
        "majority": {"class_index": majority, "class_name": CLASS_NAMES[majority],
                     "test": classification_metrics(test_labels, np.full_like(test_labels, majority), class_names=CLASS_NAMES)},
        "oracle": {
            "definition": ("each test row predicted as the majority class among the test rows of its sentence "
                           "(ties -> lowest index); uses the test labels, is the best any deterministic text "
                           "function can score, is not a model, and a model exceeding it indicates a bug"),
            "test": classification_metrics(test_labels, oracle_predictions(groups[splits["test"]], test_labels),
                                           class_names=CLASS_NAMES)},
    }


def build(*, out_dir: Path = DEFAULT_OUT, csv_path: Path | None = None, seed: int = SEED) -> dict:
    """Stage 0: label, split and describe csv5k; writes data/{dataset,rows}.json + splits.npz."""
    out = Path(out_dir) / "data"
    out.mkdir(parents=True, exist_ok=True)
    source, digest = materialize_csv(csv_path)
    rows = load_csv_rows(source)
    labels = bin_scores([row["score"] for row in rows])
    splits = grouped_split([row["group"] for row in rows], labels, seed=seed)
    group_ids = group_codes([row["group"] for row in rows])
    diag = diagnostics(rows, labels, splits)
    dataset = {
        "name": DATASET_NAME, "class_names": list(CLASS_NAMES),
        "source": {"path": str(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source), "sha256": digest},
        "n_rows": len(rows), "seed": seed, "split_protocol": "grouped by sentence (StratifiedGroupKFold, 5 folds)",
        "bin_edges": list(BIN_EDGES),
        "normalisation": "strip ' (example N)' suffix and collapse whitespace; used for grouping and model input",
        "group_encoding": "first-appearance (row-order) integer codes", "pin_fraction": PIN_FRACTION,
        "pinned_groups": [{"group": str(text), "rows": int(count)} for text, count in
                          zip(*np.unique([row["group"] for row in rows], return_counts=True)) if count >= PIN_FRACTION * len(rows)],
        "split_sizes": diag["split_sizes"], "split_sentences": diag["split_sentences"],
        "class_counts": diag["class_counts"], "diagnostics": diag,
    }
    split_of = np.empty(len(rows), dtype=object)
    for part in SPLIT_NAMES:
        split_of[splits[part]] = part
    for index, row in enumerate(rows):
        row["label"] = int(labels[index])
        row["split"] = str(split_of[index])
        row["group_id"] = int(group_ids[index])
    (out / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    (out / "rows.json").write_text(json.dumps(rows, indent=1) + "\n")
    np.savez(out / "splits.npz", fit=splits["fit"].astype(np.int64), val=splits["val"].astype(np.int64),
             test=splits["test"].astype(np.int64), labels=labels.astype(np.int64), group_ids=group_ids.astype(np.int64))
    return dataset


def load_stage0(out_dir: Path):
    """Returns (dataset_json, rows, labels, splits); splits = {"fit","val","test","group_ids"} arrays."""
    data = Path(out_dir) / "data"
    dataset = json.loads((data / "dataset.json").read_text())
    rows = json.loads((data / "rows.json").read_text())
    with np.load(data / "splits.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    labels = arrays.pop("labels")
    assert len(rows) == len(labels) == dataset["n_rows"]
    return dataset, rows, labels, arrays


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=None, help=f"source CSV (default {DEFAULT_CSV.relative_to(ROOT)})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    dataset = build(out_dir=args.out, csv_path=args.csv, seed=args.seed)
    print(json.dumps({key: dataset[key] for key in ("name", "n_rows", "split_sizes", "split_sentences",
                                                     "class_counts", "pinned_groups")}, indent=2))


if __name__ == "__main__":
    main()
