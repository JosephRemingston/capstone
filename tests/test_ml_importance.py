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
                direct = float(core.importance_scorer.model.inplace_predict(
                    np.asarray([[record.features[name] for name in FEATURE_NAMES]], dtype=np.float32))[0])
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

    def test_training_groups_and_constant_baseline(self):
        from training.train_importance import groups_for, metrics
        import numpy as np
        def row(id, worker, pair="", text=""):
            return dict(AssignmentId=id, WorkerId=worker, recAgnPairId=pair,
                        recImgPairId="", story=text or id, summary="")
        groups = groups_for([row("a", "w1"), row("b", "w2", "a"), row("c", "w2"), row("d", "w3")])
        self.assertEqual(groups[0], groups[2])
        self.assertNotEqual(groups[0], groups[3])
        self.assertIsNone(metrics(np.array([0, 1]), np.array([0.72, 0.72]))["spearman"])


if __name__ == "__main__":
    unittest.main()
