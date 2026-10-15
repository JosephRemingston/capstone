from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from training.experiments import combine, datasets, report
from training.experiments.metrics import classification_metrics, macro_f1, nll

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
    return combine.softmax(logits), logits


class BinningTests(unittest.TestCase):
    def test_bin_scores_boundaries_and_tertiles(self):
        np.testing.assert_array_equal(datasets.bin_scores([0.0, 0.39, 0.4, 0.69, 0.7, 1.0]), [0, 0, 1, 1, 2, 2])
        self.assertEqual(datasets.bin_scores([0.5]).dtype, np.int64)
        low, high = datasets.tertile_edges(np.linspace(0.2, 1.0, 301))
        self.assertLess(low, high)
        self.assertEqual(set(datasets.bin_scores(np.linspace(0.2, 1.0, 301), (low, high)).tolist()), {0, 1, 2})


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

    def test_published_split_keeps_test_trailing(self):
        labels = np.array([0, 1, 2] * 20)
        splits = datasets.published_split(labels, 7, seed=42)
        np.testing.assert_array_equal(splits["test"], np.arange(60, 67))
        self.assertEqual(len(splits["fit"]) + len(splits["val"]), 60)
        self.assertEqual(len(splits["val"]), 12)


class CsvLoaderTests(unittest.TestCase):
    def test_suffix_stripping_diagnostics_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.csv"
            tiny_csv(path)
            rows = datasets.load_csv_rows(path, datasets.REGISTRY["csv5k"])
            self.assertTrue(all("(example" not in row["text"] and row["text"].endswith(".") for row in rows))
            self.assertTrue(all(row["group"] == row["text"] for row in rows))
            self.assertEqual(len({row["text"] for row in rows}), 40)
            self.assertEqual(datasets.normalize_sentence("I like tea (example 7)."), "I like tea.")
            labels = datasets.bin_scores([row["score"] for row in rows])
            splits = datasets.grouped_split([row["group"] for row in rows], labels)
            diag = datasets.diagnostics(rows, labels, splits, class_names=("low", "mid", "high"), with_oracle=True)
            self.assertEqual(diag["rows"], len(rows))
            self.assertEqual(diag["unique_sentences"], 40)
            self.assertEqual(diag["singleton_groups"] + diag["multi_row_groups"], 40)
            self.assertEqual(sum(diag["split_sizes"].values()), len(rows))
            self.assertEqual(sum(diag["class_counts"]["all"].values()), len(rows))
            self.assertIn("macro_f1", diag["oracle"]["test"])
            self.assertIn(diag["majority"]["class_name"], ("low", "mid", "high"))
            with self.assertRaisesRegex(FileNotFoundError, "--csv5k"):
                datasets.resolve_source(datasets.REGISTRY["csv5k"], {"csv5k": str(Path(directory) / "absent.csv")})


class ClassWeightTests(unittest.TestCase):
    def test_search_never_worsens_and_identity_grid(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 3, size=300)
        probs, _ = random_probs(rng, y, 0.8)
        weights, info = combine.search_class_weights(probs, y)
        self.assertEqual(weights[2], 1.0)
        self.assertGreaterEqual(info["val_macro_f1_after"] + 1e-12, macro_f1(y, probs.argmax(axis=1)))
        self.assertAlmostEqual(info["val_macro_f1_after"], macro_f1(y, combine.apply_class_weights(probs, weights)))
        identity, info = combine.search_class_weights(probs, y, grid=(1.0,))
        self.assertEqual(identity, (1.0, 1.0, 1.0))
        np.testing.assert_array_equal(combine.apply_class_weights(probs, identity), probs.argmax(axis=1))
        self.assertEqual(info["val_macro_f1_before"], info["val_macro_f1_after"])


class MixingTests(unittest.TestCase):
    def test_average_blend_and_weight_search(self):
        rng = np.random.default_rng(3)
        y = rng.integers(0, 3, size=120)
        members = [random_probs(rng, y)[0] for _ in range(3)]
        averaged = combine.average(members)
        self.assertEqual(averaged.shape, (120, 3))
        np.testing.assert_allclose(averaged.sum(axis=1), 1.0)
        xgb, neural = random_probs(rng, y)[0], random_probs(rng, y)[0]
        self.assertEqual(combine.blend(xgb, neural, 0.25).shape, (120, 3))
        np.testing.assert_allclose(combine.blend(xgb, neural, 0.0), neural)
        np.testing.assert_allclose(combine.blend(xgb, neural, 1.0), xgb)
        w, table = combine.search_blend_weight(xgb, neural, y)
        self.assertIn(w, combine.BLEND_GRID)
        self.assertEqual([row["w"] for row in table], list(combine.BLEND_GRID))
        self.assertTrue(all({"w", "macro_f1", "nll"} <= set(row) for row in table))


class TemperatureTests(unittest.TestCase):
    def test_fit_temperature_does_not_increase_val_nll(self):
        rng = np.random.default_rng(4)
        y = rng.integers(0, 3, size=200)
        for scale in (0.3, 1.0, 4.0):
            logits = random_probs(rng, y)[1] * scale
            temperature = combine.fit_temperature(logits, y)
            self.assertTrue(0.25 <= temperature <= 10.0)
            self.assertLessEqual(nll(y, combine.softmax(logits, temperature)), nll(y, combine.softmax(logits)) + 1e-6)


class MetricTests(unittest.TestCase):
    y = np.array([0, 0, 1, 1, 2, 2, 2, 1])
    pred = np.array([0, 1, 1, 1, 2, 0, 2, 2])

    def test_keys_and_values_with_custom_names(self):
        result = classification_metrics(self.y, self.pred, class_names=("a", "b", "c"))
        self.assertEqual(set(result), {"accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "confusion_matrix", "per_class"})
        self.assertEqual(list(result["per_class"]), ["a", "b", "c"])
        self.assertAlmostEqual(result["accuracy"], 5 / 8)
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0], [0, 2, 1], [1, 0, 2]])
        self.assertEqual(result["per_class"]["c"]["support"], 3)
        self.assertAlmostEqual(result["macro_f1"], macro_f1(self.y, self.pred))

    @unittest.skipUnless(HAS_XGB, "Install requirements-ml.txt to compare with the shipped trainer")
    def test_matches_shipped_metrics_and_target(self):
        from training.train_conversational_retention import CLASSES, metrics, target
        self.assertEqual(classification_metrics(self.y, self.pred, class_names=CLASSES), metrics(self.y, self.pred))
        cases = [{"broken": "Yes", "duration": "None"}, {"broken": "No", "duration": "Short-term"},
                 {"broken": "No", "duration": "Long-term"}, {"broken": "No", "duration": "None"}]
        self.assertEqual([datasets.msc_target(row) for row in cases], [target(row) for row in cases])
        self.assertEqual([datasets.msc_target(row) for row in cases], [0, 1, 2, None])


class ReportTests(unittest.TestCase):
    def test_render_markdown_with_ok_and_failed_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tiny_csv(root / "tiny.csv", sentences=60, seed=5)
            with patch.object(datasets, "DATA_DIR", root / "materialized"):
                dataset = datasets.build("csv5k", overrides={"csv5k": str(root / "tiny.csv")}, out_dir=root / "out")
            dataset_dir = root / "out" / "csv5k"
            _, _, labels, splits = datasets.load_stage0(dataset_dir)
            rng = np.random.default_rng(6)
            xgb = dataset_dir / "xgboost"
            xgb.mkdir()
            y_val, y_test = labels[splits["val"]], labels[splits["test"]]
            np.save(xgb / "probs_val.npy", random_probs(rng, y_val)[0].astype(np.float32))
            np.save(xgb / "probs_test_fitonly.npy", random_probs(rng, y_test)[0].astype(np.float32))
            np.save(xgb / "probs_test_refit.npy", random_probs(rng, y_test)[0].astype(np.float32))
            np.save(xgb / "y_val.npy", y_val)
            np.save(xgb / "y_test.npy", y_test)
            (xgb / "grid.json").write_text(json.dumps({"trials": [{"max_depth": 3, "min_child_weight": 3, "best_iteration": 9,
                                                                   "macro_f1": 0.5, "balanced_accuracy": 0.5}],
                                                       "chosen": {"max_depth": 3, "min_child_weight": 3, "best_iteration": 9}}))
            combined = combine.build_variants(dataset_dir, resamples=20)
            self.assertEqual(combined["dataset"], dataset["name"])
            self.assertEqual(combined["variants"]["xgb_fitonly"]["status"], "ok")
            self.assertEqual(combined["variants"]["deep_asis"]["status"], "missing")
            self.assertTrue((dataset_dir / "combined.json").exists())
            summary = report.build_summary({"tiny": combined, "broken": None}, {"argv": ["run_all"], "git_head": "abc",
                                                                               "steps": [{"name": "neural", "exit_code": 1}]})
            text = report.render_markdown(summary)
            json_path, md_path = report.write(summary, root / "report")
            self.assertTrue(json_path.exists() and md_path.exists())
            self.assertIn("## tiny", text)
            self.assertIn("## broken", text)
            self.assertIn("Pipeline failed", text)
            self.assertIn("| deep_asis | — | — | — | — | — | — | — |", text)
            self.assertIn("| xgb_fitonly |", text)
            self.assertIn("failed steps: neural", text)
            self.assertIn("CSVs are template/AI-generated with synthetic scores", text)
            self.assertIn("nothing in `main/`, `artifacts/importance*` or the shipped reports changed", text)
            self.assertIn(report.ORACLE_WORDING, text)
            self.assertEqual(summary["datasets"]["broken"], {"status": "failed"})


# xgboost and torch each run in their own interpreter, as the experiment does (one native stack per process).
def run_isolated(code: str) -> None:
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(result.stderr[-2000:])


@unittest.skipUnless(HAS_XGB, "Install requirements.txt")
class XGBoostStageTests(unittest.TestCase):
    def test_grid_selects_a_trial_and_refit_uses_its_tree_count(self):
        run_isolated('''
import numpy as np
from training.experiments.xgboost_run import GRID, fit_grid, refit
rng = np.random.default_rng(0)
y = rng.integers(0, 3, size=240).astype(np.int32)
x = rng.normal(size=(240, 525)).astype(np.float32)
x[:, 0] += y
fit, val = np.arange(160), np.arange(160, 240)
trials, chosen, model = fit_grid(x, y, fit, val, class_names=["a", "b", "c"], n_jobs=1)
assert len(trials) == len(GRID) and chosen in trials
assert model.predict_proba(x[val]).shape == (80, 3)
final = refit(x, y, np.arange(240), chosen, n_jobs=1)
assert final.get_booster().num_boosted_rounds() == chosen["best_iteration"] + 1
''')


@unittest.skipUnless(HAS_TORCH, "Install requirements.txt")
class NeuralStageTests(unittest.TestCase):
    def test_short_training_returns_logits_and_bounded_temperature(self):
        run_isolated('''
import numpy as np
from training.experiments.combine import fit_temperature
from training.experiments.neural_run import logits_of, normalisation, train_head
rng = np.random.default_rng(0)
n = 120
labels = rng.integers(0, 3, size=n).astype(np.int64)
embed = rng.normal(size=(n, 384)).astype(np.float32)
embed /= np.linalg.norm(embed, axis=1, keepdims=True)
numeric = rng.normal(size=(n, 13)).astype(np.float32)
fit, val = np.arange(80), np.arange(80, n)
mean, scale = normalisation(numeric, fit)
model, info = train_head(42, embed, numeric, labels, fit, val, mean, scale, "frozen-bge-two-branch-v1",
                         max_epochs=2, patience=1)
logits = logits_of(model, embed[val], ((numeric[val] - mean) / scale).astype(np.float32))
assert logits.shape == (40, 3) and 1 <= info["best_epoch"] <= 2
assert 0.25 <= fit_temperature(logits, labels[val]) <= 10
''')


if __name__ == "__main__":
    unittest.main()
