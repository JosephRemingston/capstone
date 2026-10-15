from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from main.ensemble import apply_class_weights, average, blend, softmax
from training.ensemble import assemble, datasets, report
from training.ensemble.metrics import classification_metrics, macro_f1, nll

ROOT = Path(__file__).resolve().parents[1]
HAS_XGB = importlib.util.find_spec("xgboost") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


def tiny_csv(path: Path, *, sentences=40, seed=0) -> None:
    """Sentences repeated 1-4 times, mostly carrying the ' (example N)' artefact before the final period."""
    rng = np.random.default_rng(seed)
    lines = ["memory,importance_score"]
    for index in range(sentences):
        base = f"I plan to learn topic {index} this month"
        for repeat in range(int(rng.integers(1, 5))):
            suffix = f" (example {index * 10 + repeat})" if rng.random() < 0.9 else ""
            lines.append(f"{base}{suffix}.,{rng.uniform(0.2, 1.0):.2f}")
    path.write_text("\n".join(lines) + "\n")


def random_probs(rng, y, strength=1.5):
    logits = rng.normal(size=(len(y), 3))
    logits[np.arange(len(y)), y] += strength
    return softmax(logits), logits


class BinningTests(unittest.TestCase):
    def test_bin_scores_boundaries(self):
        np.testing.assert_array_equal(datasets.bin_scores([0.0, 0.39, 0.4, 0.69, 0.7, 1.0]), [0, 0, 1, 1, 2, 2])
        self.assertEqual(datasets.bin_scores([0.5]).dtype, np.int64)


class GroupedSplitTests(unittest.TestCase):
    def test_partition_overlap_pin_and_seed(self):
        rng = np.random.default_rng(1)
        groups = [f"g{i}" for i in range(60) for _ in range(5)] + ["big"] * 75   # "big" holds 20% of 375 rows
        labels = rng.integers(0, 3, size=len(groups))
        splits = datasets.grouped_split(groups, labels, seed=42)
        joined = np.concatenate([splits["fit"], splits["val"], splits["test"]])
        np.testing.assert_array_equal(np.sort(joined), np.arange(len(groups)))
        array = np.asarray(groups)
        for left, right in (("fit", "val"), ("fit", "test"), ("val", "test")):
            self.assertFalse(set(array[splits[left]]) & set(array[splits[right]]))
        self.assertTrue(np.isin(np.flatnonzero(array == "big"), splits["fit"]).all())
        self.assertTrue(len(splits["test"]) and len(splits["val"]))
        again = datasets.grouped_split(groups, labels, seed=42)
        for name in ("fit", "val", "test"):
            np.testing.assert_array_equal(splits[name], again[name])
        other = datasets.grouped_split(groups, labels, seed=43)
        self.assertFalse(np.array_equal(splits["test"], other["test"]))


class CsvLoaderTests(unittest.TestCase):
    def test_suffix_stripping_diagnostics_build_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tiny_csv(root / "tiny.csv")
            rows = datasets.load_csv_rows(root / "tiny.csv")
            self.assertTrue(all("(example" not in row["text"] and row["text"].endswith(".") for row in rows))
            self.assertTrue(all(row["group"] == row["text"] for row in rows))
            self.assertEqual(len({row["text"] for row in rows}), 40)
            self.assertEqual(datasets.normalize_sentence("I like tea (example 7)."), "I like tea.")
            labels = datasets.bin_scores([row["score"] for row in rows])
            splits = datasets.grouped_split([row["group"] for row in rows], labels)
            diag = datasets.diagnostics(rows, labels, splits)
            self.assertEqual(diag["rows"], len(rows))
            self.assertEqual(diag["unique_sentences"], 40)
            self.assertEqual(diag["singleton_groups"] + diag["multi_row_groups"], 40)
            self.assertEqual(sum(diag["split_sizes"].values()), len(rows))
            self.assertIn("macro_f1", diag["oracle"]["test"])
            with patch.object(datasets, "DATA_DIR", root / "copied"), patch.object(datasets, "DEFAULT_CSV", root / "copied" / "csv5k.csv"):
                dataset = datasets.build(out_dir=root / "out", csv_path=root / "tiny.csv")
                self.assertTrue((root / "copied" / "csv5k.csv").exists())
                with self.assertRaisesRegex(FileNotFoundError, "--csv"):
                    datasets.materialize_csv(root / "absent.csv")
            _, loaded_rows, loaded_labels, loaded_splits = datasets.load_stage0(root / "out")
            self.assertEqual(dataset["n_rows"], len(loaded_rows))
            self.assertEqual(set(loaded_splits), {"fit", "val", "test", "group_ids"})
            self.assertEqual(len(loaded_labels), len(rows))


class ClassWeightTests(unittest.TestCase):
    def test_search_never_worsens_and_identity_grid(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 3, size=300)
        probs, _ = random_probs(rng, y, 0.8)
        weights, info = assemble.search_class_weights(probs, y)
        self.assertEqual(weights[2], 1.0)
        self.assertGreaterEqual(info["val_macro_f1_after"] + 1e-12, macro_f1(y, probs.argmax(axis=1)))
        self.assertAlmostEqual(info["val_macro_f1_after"], macro_f1(y, apply_class_weights(probs, weights)))
        identity, info = assemble.search_class_weights(probs, y, grid=(1.0,))
        self.assertEqual(identity, (1.0, 1.0, 1.0))
        np.testing.assert_array_equal(apply_class_weights(probs, identity), probs.argmax(axis=1))
        self.assertEqual(info["val_macro_f1_before"], info["val_macro_f1_after"])


class MixingTests(unittest.TestCase):
    def test_average_blend_and_weight_search(self):
        rng = np.random.default_rng(3)
        y = rng.integers(0, 3, size=120)
        members = [random_probs(rng, y)[0] for _ in range(3)]
        averaged = average(members)
        self.assertEqual(averaged.shape, (120, 3))
        np.testing.assert_allclose(averaged.sum(axis=1), 1.0)
        xgb, neural = random_probs(rng, y)[0], random_probs(rng, y)[0]
        np.testing.assert_allclose(blend(xgb, neural, 0.0), neural)
        np.testing.assert_allclose(blend(xgb, neural, 1.0), xgb)
        w, table = assemble.search_blend_weight(xgb, neural, y)
        self.assertIn(w, assemble.BLEND_GRID)
        self.assertEqual([row["w"] for row in table], list(assemble.BLEND_GRID))


class TemperatureTests(unittest.TestCase):
    def test_fit_temperature_does_not_increase_val_nll(self):
        rng = np.random.default_rng(4)
        y = rng.integers(0, 3, size=200)
        for scale in (0.3, 1.0, 4.0):
            logits = random_probs(rng, y)[1] * scale
            temperature = assemble.fit_temperature(logits, y)
            self.assertTrue(0.25 <= temperature <= 10.0)
            self.assertLessEqual(nll(y, softmax(logits, temperature)), nll(y, softmax(logits)) + 1e-6)


class MetricTests(unittest.TestCase):
    def test_keys_and_values_with_custom_names(self):
        y, pred = np.array([0, 0, 1, 1, 2, 2, 2, 1]), np.array([0, 1, 1, 1, 2, 0, 2, 2])
        result = classification_metrics(y, pred, class_names=("a", "b", "c"))
        self.assertEqual(set(result), {"accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "confusion_matrix", "per_class"})
        self.assertAlmostEqual(result["accuracy"], 5 / 8)
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0], [0, 2, 1], [1, 0, 2]])
        self.assertAlmostEqual(result["macro_f1"], macro_f1(y, pred))


class ReportTests(unittest.TestCase):
    def test_tracked_markdown_is_rendered_from_the_tracked_json(self):
        summary = json.loads((report.REPORT_DIR / f"{report.REPORT_NAME}.json").read_text())
        text = report.render_markdown(summary)
        self.assertEqual(text, (report.REPORT_DIR / f"{report.REPORT_NAME}.md").read_text())
        self.assertIn("**Ensemble + thresholds**", text)
        self.assertEqual(summary["serving_checks"]["live_test"]["class_mismatches"], 0)


# One native ML stack per interpreter, as the training does.
def run_isolated(code: str) -> None:
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(result.stderr[-2000:])


class ProcessIsolationTests(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "Install requirements.txt")
    def test_neural_stage_never_loads_xgboost(self):
        run_isolated("import sys, training.ensemble.neural_stage; assert 'xgboost' not in sys.modules")

    @unittest.skipUnless(HAS_XGB, "Install requirements.txt")
    def test_xgboost_stage_and_assemble_never_load_torch(self):
        run_isolated("import sys, training.ensemble.xgboost_stage, training.ensemble.assemble; assert 'torch' not in sys.modules")


@unittest.skipUnless(HAS_XGB, "Install requirements.txt")
class XGBoostStageTests(unittest.TestCase):
    def test_grid_winner_is_saved_cut_to_its_best_iteration(self):
        run_isolated('''
import numpy as np
from training.ensemble.xgboost_stage import GRID, fit_grid, sliced_booster
rng = np.random.default_rng(0)
y = rng.integers(0, 3, size=240).astype(np.int32)
x = rng.normal(size=(240, 525)).astype(np.float32)
x[:, 0] += y
fit, val = np.arange(160), np.arange(160, 240)
trials, chosen, model = fit_grid(x, y, fit, val, class_names=["a", "b", "c"], n_jobs=1)
assert len(trials) == len(GRID) and chosen in trials
booster = sliced_booster(model, chosen["best_iteration"] + 1)
assert booster.num_boosted_rounds() == chosen["best_iteration"] + 1
assert np.array_equal(booster.inplace_predict(x[val]).astype(np.float32), model.predict_proba(x[val]).astype(np.float32))
''')


@unittest.skipUnless(HAS_TORCH, "Install requirements.txt")
class NeuralStageTests(unittest.TestCase):
    def test_short_training_and_onnx_export(self):
        run_isolated('''
import tempfile
from pathlib import Path
import numpy as np
from training.ensemble.neural_stage import export_onnx, logits_of, normalisation, train_head
rng = np.random.default_rng(0)
n = 120
labels = rng.integers(0, 3, size=n).astype(np.int64)
embed = rng.normal(size=(n, 384)).astype(np.float32)
embed /= np.linalg.norm(embed, axis=1, keepdims=True)
numeric = rng.normal(size=(n, 13)).astype(np.float32)
fit, val = np.arange(80), np.arange(80, n)
mean, scale = normalisation(numeric, fit)
model, info = train_head(42, embed, numeric, labels, fit, val, mean, scale, max_epochs=2, patience=1)
scaled = ((numeric[val] - mean) / scale).astype(np.float32)
assert logits_of(model, embed[val], scaled).shape == (40, 3) and 1 <= info["best_epoch"] <= 2
with tempfile.TemporaryDirectory() as directory:
    assert export_onnx(model, Path(directory) / "head.onnx", embed[val], scaled) < 1e-4
''')


if __name__ == "__main__":
    unittest.main()
