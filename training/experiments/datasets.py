"""Stage 0: dataset registry, loaders, binning, grouped split and diagnostics.

    python -m training.experiments.datasets build --dataset csv5k --out artifacts/experiments/retention_datasets

Writes ``<out>/<dataset>/data/{dataset.json, rows.json, splits.npz}``. The row order of
``rows.json`` is the index for every per-row array written by later stages.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from training.experiments.metrics import classification_metrics, majority_class

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "experiments"            # CSV copies, git-ignored
MSC_DIR = ROOT / "data" / "personal_facts_msc"
# copy of training/train_conversational_retention.py:34-37 (checksums only; no download here)
MSC_FILES = {
    "train": ("f7f7fab70fcfc949abc743480fbf189d5956fc2d05cbd0628147aa76d1551b8b", "train-00000-of-00001.parquet"),
    "test": ("91c4c166f63aa6b747604774a04d01c5b4fd5a8deedca88797a339e087640aaa", "test-00000-of-00001.parquet"),
}
EXAMPLE_SUFFIX = re.compile(r"\s*\(example \d+\)")   # NOT end-anchored: suffix sits before the final period
BIN_EDGES = (0.4, 0.7)                               # low <0.4, mid [0.4,0.7), high >=0.7
PIN_FRACTION = 0.05                                  # groups holding >=5% of rows go to fit (label-blind, pre-registered)
SEED = 42
SPLIT_NAMES = ("fit", "val", "test")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str                                  # "csv" | "msc"
    class_names: tuple[str, str, str]
    default_path: Path | None = None
    flag: str | None = None                    # CLI override flag, named in FileNotFoundError
    split: str = "grouped"                     # "grouped" (by sentence) | "published" (MSC test + shipped carve-out)


REGISTRY = {
    "csv5k": DatasetSpec("csv5k", "csv", ("low", "mid", "high"),
                         Path.home() / "Downloads" / "cognimem_memory_importance_5000.csv", "--csv5k"),
    "csv10k": DatasetSpec("csv10k", "csv", ("low", "mid", "high"),
                          Path.home() / "Downloads" / "cognimem_10000_memory_importance.csv", "--csv10k"),
    "msc": DatasetSpec("msc", "msc", ("invalid", "short_term", "long_term"), split="published"),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_source(spec: DatasetSpec, overrides: dict | None = None) -> Path:
    override = (overrides or {}).get(spec.name)
    path = Path(override).expanduser() if override else spec.default_path
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{spec.name}: source CSV not found at {path}; pass {spec.flag} PATH")
    return path


def materialize_csv(spec: DatasetSpec, source: Path) -> tuple[Path, str]:
    """Copy the CSV into DATA_DIR (once per content) so a run never depends on ~/Downloads."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / f"{spec.name}.csv"
    digest = sha256_file(source)
    if not target.exists() or sha256_file(target) != digest:
        shutil.copyfile(source, target)
    return target, digest


def normalize_sentence(text: str) -> str:
    """Strip the ' (example N)' generation artefact; used for BOTH grouping and model input."""
    return re.sub(r"\s+", " ", EXAMPLE_SUFFIX.sub("", text)).strip()


def load_csv_rows(path: Path, spec: DatasetSpec) -> list[dict]:
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


def msc_target(row: dict) -> int | None:
    # copy of training/train_conversational_retention.py:63-70
    if row["broken"] == "Yes":
        return 0
    if row["broken"] == "No" and row["duration"] == "Short-term":
        return 1
    if row["broken"] == "No" and row["duration"] == "Long-term":
        return 2
    return None


def load_msc_rows(split: str) -> tuple[list[dict], int, str]:
    """Usable MSC rows in parquet order (same omission rule as the shipped prepare loops)."""
    expected, _ = MSC_FILES[split]
    path = MSC_DIR / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run the shipped training once to download it")
    digest = sha256_file(path)
    if digest != expected:   # copy of training/train_conversational_retention.py:58-59
        raise ValueError(f"Unexpected checksum for {split} dataset")
    rows, omitted = [], 0
    for raw in pq.read_table(path).to_pylist():
        label = msc_target(raw)
        if label is None or not raw["text"].strip():   # train_conversational_retention.py:80-83
            omitted += 1
            continue
        rows.append({"text": raw["text"], "score": None, "group": raw["text"], "label": label})
    return rows, omitted, digest


def bin_scores(scores, edges=BIN_EDGES) -> np.ndarray:
    return np.searchsorted(np.asarray(edges, dtype=float), np.asarray(scores, dtype=float), side="right").astype(np.int64)


def tertile_edges(scores) -> tuple[float, float]:
    low, high = np.quantile(np.asarray(scores, dtype=float), [1 / 3, 2 / 3])
    return float(low), float(high)


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

    StratifiedGroupKFold's assignment depends on how groups are numbered; the plan's expected partitions
    (A 3,200/792/1,008, B 6,578/1,521/1,901) were computed with row-order codes. np.unique's sorted-text
    codes give an equally valid but different partition, so they are not used.
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


def published_split(labels_train, n_test: int, *, seed: int = SEED) -> dict:
    """The shipped MSC protocol; fit/val keep train_test_split's order because row order feeds the trainers."""
    labels_train = np.asarray(labels_train)
    fit, val = train_test_split(np.arange(len(labels_train)), test_size=0.2,
                                stratify=labels_train, random_state=seed)   # train_conversational_retention.py:113-115
    test = np.arange(len(labels_train), len(labels_train) + n_test)
    splits = {"fit": np.asarray(fit), "val": np.asarray(val), "test": test}
    _check_partition(splits, np.arange(len(labels_train) + n_test))
    return splits


def oracle_predictions(groups, labels) -> np.ndarray:
    """Majority label among the given rows of each group (ties -> lowest index), per row."""
    groups, labels = np.asarray(groups), np.asarray(labels)
    _, inverse = np.unique(groups, return_inverse=True)
    counts = np.zeros((inverse.max() + 1, 3), dtype=np.int64)
    np.add.at(counts, (inverse, labels), 1)
    return counts.argmax(axis=1)[inverse]


def class_counts(labels, class_names) -> dict:
    return {name: int((np.asarray(labels) == i).sum()) for i, name in enumerate(class_names)}


def diagnostics(rows: list[dict], labels, splits: dict, *, class_names, with_oracle: bool) -> dict:
    labels = np.asarray(labels)
    groups = np.asarray([row["group"] for row in rows])
    scores = np.asarray([row["score"] if row["score"] is not None else np.nan for row in rows], dtype=float)
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    multi = np.flatnonzero(counts > 1)
    disagree = sum(1 for g in multi if len(set(labels[inverse == g].tolist())) > 1)
    score_std = [float(np.std(scores[inverse == g])) for g in multi] if np.isfinite(scores).all() else []
    largest = int(counts.argmax())
    tune_labels = labels[np.concatenate([splits["fit"], splits["val"]])]
    majority = majority_class(tune_labels)
    test_labels = labels[splits["test"]]
    result = {
        "rows": len(rows), "unique_sentences": int(len(unique)),
        "singleton_groups": int((counts == 1).sum()), "multi_row_groups": int(len(multi)),
        "largest_group": {"text": str(unique[largest]), "rows": int(counts[largest])},
        "disagreeing_multi_groups": int(disagree),
        "disagreeing_multi_group_share": float(disagree / len(multi)) if len(multi) else None,
        "mean_within_group_score_std": float(np.mean(score_std)) if score_std else None,
        "class_counts": {"all": class_counts(labels, class_names),
                         **{name: class_counts(labels[splits[name]], class_names) for name in SPLIT_NAMES}},
        "split_sizes": {name: int(len(splits[name])) for name in SPLIT_NAMES},
        "split_sentences": {name: int(len(set(groups[splits[name]]))) for name in SPLIT_NAMES},
        "majority": {"class_index": majority, "class_name": class_names[majority],
                     "test": classification_metrics(test_labels, np.full_like(test_labels, majority), class_names=class_names)},
        "oracle": None,
    }
    if with_oracle:
        pred = oracle_predictions(groups[splits["test"]], test_labels)
        result["oracle"] = {
            "definition": ("each test row predicted as the majority class among the test rows of its sentence "
                           "(ties -> lowest index); uses the test labels, is the best any deterministic text "
                           "function can score, is not a model, and a model exceeding it indicates a bug"),
            "test": classification_metrics(test_labels, pred, class_names=class_names)}
    return result


def build(name: str, *, overrides: dict | None = None, out_dir: Path, seed: int = SEED,
          bin_edges=BIN_EDGES) -> dict:
    """Stage 0: materialise, label, split and describe one dataset; writes data/{dataset,rows}.json + splits.npz."""
    spec = REGISTRY[name]
    out = Path(out_dir) / name / "data"
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    if spec.kind == "csv":
        source = resolve_source(spec, overrides)
        copied, digest = materialize_csv(spec, source)
        rows = load_csv_rows(copied, spec)
        scores = np.asarray([row["score"] for row in rows])
        labels = bin_scores(scores, bin_edges)
        tertiles = tertile_edges(scores)
        arrays["tertile_labels"] = bin_scores(scores, tertiles)
        splits = grouped_split([row["group"] for row in rows], labels, seed=seed)
        source_info = {"path": str(source), "sha256": digest, "materialized": str(copied.relative_to(ROOT) if copied.is_relative_to(ROOT) else copied)}
        omitted = 0
    else:
        train_rows, omitted_train, train_sha = load_msc_rows("train")
        test_rows, omitted_test, test_sha = load_msc_rows("test")
        rows = train_rows + test_rows
        labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
        tertiles = None
        splits = published_split(labels[:len(train_rows)], len(test_rows), seed=seed)
        source_info = {"train": {"path": str(MSC_DIR / "train.parquet"), "sha256": train_sha},
                       "test": {"path": str(MSC_DIR / "test.parquet"), "sha256": test_sha}}
        omitted = omitted_train + omitted_test
    groups = [row["group"] for row in rows]
    group_ids = group_codes(groups)
    pinned = [{"group": text, "rows": count} for text, count in Counter(groups).items()
              if count >= PIN_FRACTION * len(rows)] if spec.kind == "csv" else []
    diag = diagnostics(rows, labels, splits, class_names=spec.class_names, with_oracle=spec.kind == "csv")
    dataset = {
        "name": name, "kind": spec.kind, "class_names": list(spec.class_names), "source": source_info,
        "n_rows": len(rows), "omitted_rows": omitted, "seed": seed, "split_protocol": spec.split,
        "bin_edges": list(bin_edges) if spec.kind == "csv" else None,
        "tertile_edges": list(tertiles) if tertiles else None,
        "normalisation": ("strip ' (example N)' suffix and collapse whitespace; used for grouping and model input"
                          if spec.kind == "csv" else "none (parquet text as shipped)"),
        "grouped_by_sentence": spec.kind == "csv",
        "group_encoding": "first-appearance (row-order) integer codes" if spec.kind == "csv" else None,
        "pin_fraction": PIN_FRACTION if spec.kind == "csv" else None, "pinned_groups": pinned,
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
             test=splits["test"].astype(np.int64), labels=labels.astype(np.int64),
             group_ids=group_ids.astype(np.int64), **arrays)
    return dataset


def load_stage0(dataset_dir: Path):
    """Returns (dataset_json, rows, labels, splits); splits = {"fit","val","test"} index arrays + extra npz keys."""
    data = Path(dataset_dir) / "data"
    dataset = json.loads((data / "dataset.json").read_text())
    rows = json.loads((data / "rows.json").read_text())
    with np.load(data / "splits.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    labels = arrays.pop("labels")
    assert len(rows) == len(labels) == dataset["n_rows"]
    return dataset, rows, labels, arrays


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("build")
    cmd.add_argument("--dataset", choices=sorted(REGISTRY), required=True)
    cmd.add_argument("--csv5k", type=Path)
    cmd.add_argument("--csv10k", type=Path)
    cmd.add_argument("--out", type=Path, default=ROOT / "artifacts" / "experiments" / "retention_datasets")
    cmd.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    overrides = {key: value for key, value in (("csv5k", args.csv5k), ("csv10k", args.csv10k)) if value}
    dataset = build(args.dataset, overrides=overrides, out_dir=args.out, seed=args.seed)
    print(json.dumps({key: dataset[key] for key in ("name", "n_rows", "split_sizes", "split_sentences",
                                                     "class_counts", "pinned_groups")}, indent=2))


if __name__ == "__main__":
    main()
