from __future__ import annotations

import importlib.util
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from main import MemoryCore, MemoryInput
from main.__main__ import main as cli_main
from main.domain.classification import MemoryClassifier
from main.ml import MLFeatureExtractor
from main.neural import DEFAULT_NEURAL_DIR, NeuralImportanceScorer, create_head


HAS_TORCH = importlib.util.find_spec("torch") is not None


class FakeEmbedder:
    dimension = 384

    def __init__(self, model_id):
        self.model_id = model_id
        self.seen = []

    def passages(self, texts):
        self.seen.extend(texts)
        return [[1 / math.sqrt(384)] * 384 for _ in texts]


@unittest.skipUnless(HAS_TORCH, "Install requirements-neural.txt")
class NeuralImportanceTests(unittest.TestCase):
    def test_layer_shapes_and_deterministic_scoring(self):
        # Run training-library checks separately from XGBoost's native runtime.
        code = '''
import numpy as np
import torch
import json
from main.neural import DEFAULT_NEURAL_DIR, create_head
from onnxruntime import InferenceSession
metadata = json.loads((DEFAULT_NEURAL_DIR / "metadata.json").read_text())
model = create_head(metadata["architecture"])
assert tuple(model(torch.zeros(2, 384), torch.zeros(2, 13)).shape) == (2, 3)
assert sum(p.numel() for p in create_head("frozen-bge-deep-head-v1").parameters()) > sum(p.numel() for p in create_head("frozen-bge-two-branch-v1").parameters())
try:
    model(torch.zeros(2, 383), torch.zeros(2, 13))
except ValueError:
    pass
else:
    raise AssertionError("wrong embedding dimension accepted")
model.load_state_dict(torch.load(DEFAULT_NEURAL_DIR / "model.pt", map_location="cpu", weights_only=True))
model.eval()
embedding = np.full((1, 384), 1 / np.sqrt(384), dtype=np.float32)
numeric = np.zeros((1, 13), dtype=np.float32)
session = InferenceSession(str(DEFAULT_NEURAL_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
onnx_logits = session.run(["logits"], {"embedding": embedding, "numeric": numeric})[0]
with torch.inference_mode():
    torch_logits = model(torch.from_numpy(embedding), torch.from_numpy(numeric)).numpy()
np.testing.assert_allclose(onnx_logits, torch_logits, atol=1e-5)
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                text=True, timeout=30, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((DEFAULT_NEURAL_DIR / "metadata.json").read_text())
        fake = FakeEmbedder(metadata["encoder_id"])
        scorer = NeuralImportanceScorer(embedder=fake)
        core = MemoryCore(classifier=MemoryClassifier(legacy=True),
                          feature_extractor=MLFeatureExtractor(),
                          importance_scorer=scorer)
        incoming = MemoryInput(content="I prefer concise replies.", user_id="u", session_id="s")
        first = core.process(incoming)
        second = core.process(incoming)
        self.assertEqual(first.importance_score, second.importance_score)
        self.assertGreaterEqual(first.importance_score, 0)
        self.assertLessEqual(first.importance_score, 1)
        self.assertEqual(first.source_metadata["importance_model"]["type"], "neural")
        self.assertEqual(first.tier, core.lifecycle_manager.assign_tier(first, first.importance_score))

    def test_redaction_happens_before_encoder(self):
        metadata = json.loads((DEFAULT_NEURAL_DIR / "metadata.json").read_text())
        fake = FakeEmbedder(metadata["encoder_id"])
        scorer = NeuralImportanceScorer(embedder=fake)
        core = MemoryCore(classifier=MemoryClassifier(legacy=True),
                          feature_extractor=MLFeatureExtractor(), importance_scorer=scorer)
        record = core.process(MemoryInput(content="Email me at example@example.com.",
                                          user_id="u", session_id="s"))
        self.assertNotIn("example@example.com", fake.seen[0])
        self.assertIn("[REDACTED_EMAIL]", fake.seen[0])
        self.assertTrue(record.source_metadata["privacy"]["redacted"])

    def test_artifact_checks_and_missing_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                NeuralImportanceScorer(directory)
            path = Path(directory)
            (path / "metadata.json").write_bytes((DEFAULT_NEURAL_DIR / "metadata.json").read_bytes())
            (path / "model.onnx").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                NeuralImportanceScorer(directory)
        with patch("main.neural._torch", side_effect=ValueError("Install requirements-neural.txt")):
            with self.assertRaisesRegex(ValueError, "requirements-neural"):
                create_head()
            metadata = json.loads((DEFAULT_NEURAL_DIR / "metadata.json").read_text())
            scorer = NeuralImportanceScorer(embedder=FakeEmbedder(metadata["encoder_id"]))
            self.assertTrue(math.isfinite(scorer.score_text("hello", {
                name: 0.0 for name in metadata["feature_names"]
            })))

    def test_cli_selects_neural(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with redirect_stdout(output):
                status = cli_main(["--store", str(Path(directory) / "store.jsonl"),
                                   "process", "I live in Bengaluru.", "--user-id", "u",
                                   "--session-id", "s", "--scorer", "neural", "--no-save"])
            self.assertEqual(status, 0)
            record = json.loads(output.getvalue())
            self.assertEqual(record["source_metadata"]["importance_model"]["type"], "neural")
            self.assertTrue(math.isfinite(record["importance_score"]))


if __name__ == "__main__":
    unittest.main()
