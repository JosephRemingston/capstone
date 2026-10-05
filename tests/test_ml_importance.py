from __future__ import annotations

import importlib.util
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from main import MemoryCore, MemoryInput, MLImportanceScorer
from main.__main__ import main as cli_main
from main.ml import DEFAULT_MODEL_DIR

HAS_ML = importlib.util.find_spec("xgboost") is not None


@unittest.skipUnless(HAS_ML, "Install requirements-ml.txt for ML integration tests")
class MLIntegrationTests(unittest.TestCase):
    def test_core_native_model_parity_and_metadata(self):
        import numpy as np
        from main.ml import FEATURE_NAMES
        core = MemoryCore.with_ml()
        for text in ("hello", "I prefer concise Python explanations.", "Yesterday my child was born and it changed my life."):
            with self.subTest(text=text):
                record = core.process(MemoryInput(content=text, user_id="u1", session_id="s1"))
                raw = core.importance_scorer.model.inplace_predict(
                    np.asarray([[record.features[name] for name in FEATURE_NAMES]], dtype=np.float32))[0]
                if core.importance_scorer.metadata.get("output_type") == "multiclass_probability":
                    direct = float(np.dot(raw, core.importance_scorer.metadata["score_values"]))
                else:
                    direct = float(raw)
                self.assertAlmostEqual(record.importance_score, max(0, min(direct, 1)), places=4)
                self.assertTrue(math.isfinite(record.importance_score))
                self.assertEqual(record.source_metadata["importance_model"]["type"], "xgboost")
                self.assertEqual(record.tier, core.lifecycle_manager.assign_tier(record, record.importance_score))

    def test_invalid_artifact_and_features_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                MLImportanceScorer(directory)
            path = Path(directory)
            (path / "metadata.json").write_bytes((DEFAULT_MODEL_DIR / "metadata.json").read_bytes())
            (path / "model.ubj").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                MLImportanceScorer(directory)
        scorer = MLImportanceScorer()
        with self.assertRaisesRegex(ValueError, "MLFeatureExtractor"):
            scorer.score({"recency": 1})
        core = MemoryCore.with_ml()
        record = core.process(MemoryInput(content="My project is CogniMem.", user_id="u", session_id="s"))
        record.features["entity_density"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            scorer.score(record.features)

    def test_cli_persists_ml_score_and_rejects_bad_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            store = str(Path(directory) / "store.jsonl")
            output = io.StringIO()
            with redirect_stdout(output):
                status = cli_main(["--store", store, "process", "I prefer clear examples.",
                                   "--user-id", "u", "--session-id", "s", "--scorer", "xgboost"])
            self.assertEqual(status, 0)
            record = json.loads(output.getvalue())
            self.assertEqual(record, json.loads(Path(store).read_text()))
            self.assertEqual(record["source_metadata"]["importance_model"]["type"], "xgboost")
            with redirect_stdout(io.StringIO()):
                status = cli_main(["process", "hello", "--user-id", "u", "--session-id", "s",
                                   "--model-dir", directory, "--no-save"])
            self.assertEqual(status, 2)

    def test_training_target_mapping(self):
        from training.train_conversational_retention import target
        self.assertEqual(target({"broken": "Yes", "duration": "None"}), 0)
        self.assertEqual(target({"broken": "No", "duration": "Short-term"}), 1)
        self.assertEqual(target({"broken": "No", "duration": "Long-term"}), 2)
        self.assertIsNone(target({"broken": "No", "duration": "None"}))


if __name__ == "__main__":
    unittest.main()
