"""The default importance scorer: 25 % csv5k ensemble + 75 % heuristic (main/ensemble.py)."""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from main import MemoryCore, MemoryInput
from main.__main__ import main as cli_main
from main.domain.scoring import ImportanceScorer
from main.ensemble import (BGE_CACHE_DIR, CLASS_ORDER, DEFAULT_ENSEMBLE_DIR, MODEL_WEIGHT, SMALL_TALK_PENALTY,
                           EnsembleModel, HybridImportanceScorer, _load_model, build_scorer, default_embedder,
                           default_importance_scorer, load_model, selected_scorer)

HAS_MODEL = (DEFAULT_ENSEMBLE_DIR / "metadata.json").is_file()
HAS_BGE = any(BGE_CACHE_DIR.glob("models--*bge-small-en-v1.5*"))
NEEDS_MODEL = unittest.skipUnless(HAS_MODEL, "train the model with `python -m training.ensemble.run`")
NEEDS_BGE = unittest.skipUnless(HAS_MODEL and HAS_BGE, "needs the trained model and the cached BGE encoder")


class FakeEmbedder:
    """Deterministic 384-d unit vectors that claim the model's encoder id, and record what they were shown."""
    dimension = 384

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.seen: list[str] = []

    def passages(self, texts):
        self.seen += texts
        vectors = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
            vector = np.random.default_rng(seed).normal(size=384)
            vectors.append((vector / np.linalg.norm(vector)).tolist())
        return vectors


def fake_scorer() -> HybridImportanceScorer:
    model = load_model()
    return HybridImportanceScorer(model, embedder=FakeEmbedder(model.encoder_id))


@NEEDS_MODEL
class ArtifactValidationTests(unittest.TestCase):
    def copy(self) -> Path:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory())) / "model"
        shutil.copytree(DEFAULT_ENSEMBLE_DIR, directory)
        return directory

    def rewrite(self, directory: Path, change) -> None:
        metadata = json.loads((directory / "metadata.json").read_text())
        change(metadata)
        (directory / "metadata.json").write_text(json.dumps(metadata))

    def test_valid_copy_loads(self):
        model = EnsembleModel(self.copy())
        self.assertEqual(model.class_values, tuple(model.metadata["class_values"]))
        self.assertEqual(model.metadata["xgboost"]["rounds"], model.metadata["xgboost"]["best_iteration"] + 1)

    def test_missing_directory_names_the_training_command_and_opt_out(self):
        with self.assertRaisesRegex(ValueError, "training.ensemble.run.*COGNIMEM_SCORER=heuristic"):
            EnsembleModel(Path(self.enterContext(tempfile.TemporaryDirectory())) / "absent")

    def test_tampered_or_incompatible_artifacts_fail_explicitly(self):
        tampered = self.copy()
        data = bytearray((tampered / "xgboost.ubj").read_bytes())
        data[-1] ^= 1
        (tampered / "xgboost.ubj").write_bytes(bytes(data))
        with self.assertRaisesRegex(ValueError, "checksum mismatch: xgboost.ubj"):
            EnsembleModel(tampered)
        missing = self.copy()
        (missing / "deep_seed_43.onnx").unlink()
        with self.assertRaisesRegex(ValueError, "Cannot load importance model"):
            EnsembleModel(missing)
        for change, message in ((lambda m: m.update(class_order=["invalid", "short_term", "long_term"]), "class order"),
                                (lambda m: m["neural"]["members"][0].update(temperature=0.0), "Temperatures"),
                                (lambda m: m.update(class_values=[0.8, 0.5, 0.3]), "class values"),
                                (lambda m: m["decision"].update(class_weights=[1.0, -0.7, 1.0]), "Class weights"),
                                (lambda m: m["xgboost"].update(rounds=329), "XGBoost member")):
            with self.subTest(message=message):
                broken = self.copy()
                self.rewrite(broken, change)
                with self.assertRaisesRegex(ValueError, message):
                    EnsembleModel(broken)

    def test_missing_encoder_cache_gives_the_download_command_and_opt_out(self):
        with patch("main.ensemble.BGE_CACHE_DIR", Path(self.enterContext(tempfile.TemporaryDirectory()))):
            from main.ensemble import local_embedder
            with self.assertRaisesRegex(ValueError, "not cached.*FastEmbedder\\(\\).*COGNIMEM_SCORER=heuristic"):
                local_embedder()

    def test_metadata_must_name_the_encoder(self):
        broken = self.copy()
        self.rewrite(broken, lambda m: m["neural"].pop("encoder_id"))
        with self.assertRaisesRegex(ValueError, "Cannot load importance model"):
            EnsembleModel(broken)

    def test_encoder_must_match(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            HybridImportanceScorer(load_model(), embedder=FakeEmbedder("some-other-encoder"))


@NEEDS_MODEL
class HybridScoringTests(unittest.TestCase):
    def test_blend_formula_and_small_talk_penalty(self):
        core = MemoryCore(importance_scorer=fake_scorer())
        model = load_model()
        for text, small_talk in (("I prefer Python examples with clear tests.", False), ("hello", True)):
            with self.subTest(text=text):
                record = core.process(MemoryInput(text, "u", "s"))
                stamp = record.source_metadata["importance_model"]
                self.assertEqual(stamp["type"], "hybrid")
                self.assertEqual(stamp["small_talk_penalty"], small_talk)
                value = model.class_values[CLASS_ORDER.index(stamp["model_class"])]
                expected_model = value * (SMALL_TALK_PENALTY if small_talk else 1.0)
                self.assertAlmostEqual(stamp["model_score"], round(expected_model, 4))
                self.assertEqual(stamp["heuristic_score"], ImportanceScorer().score(record.features))
                self.assertEqual(record.importance_score,
                                 round(MODEL_WEIGHT * expected_model + (1 - MODEL_WEIGHT) * stamp["heuristic_score"], 4))
                self.assertEqual(stamp["model_sha256"], model.sha256)

    def test_redaction_happens_before_the_encoder(self):
        scorer = fake_scorer()
        MemoryCore(importance_scorer=scorer).process(MemoryInput("Email me at alice@example.com about tea.", "u", "s"))
        self.assertTrue(scorer.embedder.seen)
        self.assertFalse(any("alice@example.com" in text for text in scorer.embedder.seen))

    def test_provenance_overwrites_caller_metadata(self):
        record = MemoryCore(importance_scorer=fake_scorer()).process(
            MemoryInput("I like tea.", "u", "s", metadata={"importance_model": "spoofed"}))
        self.assertEqual(record.source_metadata["importance_model"]["type"], "hybrid")
        heuristic = MemoryCore.with_heuristic().process(MemoryInput("I like tea.", "u", "s", metadata={"importance_model": "spoofed"}))
        self.assertEqual(heuristic.source_metadata["importance_model"], {"type": "heuristic"})

    def test_text_is_required(self):
        with self.assertRaises(ValueError):
            fake_scorer().score({})

    def test_model_bundle_is_cached_per_directory(self):
        self.assertIs(load_model(), load_model())
        first = load_model()
        _load_model.cache_clear()
        self.assertIsNot(load_model(), first)


class ChosenFormulaTests(unittest.TestCase):
    """The user's choice, pinned literally: a change to any of these numbers must fail a test."""

    def test_weights_and_penalty(self):
        self.assertEqual(MODEL_WEIGHT, 0.25)
        self.assertEqual(SMALL_TALK_PENALTY, 0.25)
        self.assertEqual(SMALL_TALK_PENALTY, ImportanceScorer().temporary_penalty)

    @NEEDS_MODEL
    def test_class_values_and_a_hand_computed_score(self):
        self.assertEqual(load_model().class_values, (0.3102, 0.5792, 0.8372))
        scorer = fake_scorer()
        with patch.object(scorer, "predict", return_value=("high", np.array([0.0, 0.1, 0.9]))):
            score, stamp = scorer.assess("hello", {"category_temporary": 1.0})   # heuristic of these features: 0.0
            self.assertEqual((score, stamp["model_score"]), (round(0.25 * 0.8372 * 0.25, 4), round(0.8372 * 0.25, 4)))
            score, _ = scorer.assess("I like tea.", {"category_preference": 1.0, "preference_signal": 1.0})
            self.assertEqual(score, round(0.25 * 0.8372 + 0.75 * 0.5, 4))   # heuristic: 0.38 + 0.12


class SettingsScorerTests(unittest.TestCase):
    def test_api_settings_read_the_scorer_from_the_environment(self):
        from api.config import Settings
        base = {"COGNIMEM_JWT_SECRET": "x" * 48}
        for value, expected in (("", "hybrid"), ("heuristic", "heuristic"), ("HYBRID", "hybrid")):
            with self.subTest(value=value), patch.dict(os.environ, {**base, "COGNIMEM_SCORER": value}):
                self.assertEqual(Settings.from_env().scorer, expected)
        with patch.dict(os.environ, {**base, "COGNIMEM_SCORER": "xgboost"}), self.assertRaises(ValueError):
            Settings.from_env()


class ScorerSelectionTests(unittest.TestCase):
    def test_names_are_validated(self):
        self.assertEqual(selected_scorer("HYBRID"), "hybrid")
        with patch.dict(os.environ, {"COGNIMEM_SCORER": " "}):
            self.assertEqual(selected_scorer(), "hybrid")   # blank means unset, as for every COGNIMEM_* variable
        with self.assertRaisesRegex(ValueError, "COGNIMEM_SCORER must be one of"):
            selected_scorer("xgboost")
        with patch.dict(os.environ, {"COGNIMEM_SCORER": "heuristic"}):
            self.assertIsInstance(default_importance_scorer(), ImportanceScorer)
            self.assertIsInstance(MemoryCore().importance_scorer, ImportanceScorer)
        with self.assertRaisesRegex(ValueError, "requires the hybrid scorer"):
            build_scorer("heuristic", model_dir="artifacts/importance_ensemble")

    def test_cli_rejects_a_model_directory_with_the_heuristic(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as output:
            code = cli_main(["--store", str(Path(directory) / "s.jsonl"), "process", "I like tea.", "--user-id", "u",
                             "--session-id", "s", "--scorer", "heuristic", "--model-dir", "artifacts/importance_ensemble",
                             "--no-save"])
        self.assertEqual(code, 2)
        self.assertIn("invalid_input", output.getvalue())


@NEEDS_BGE
class DefaultHybridTests(unittest.TestCase):
    """The shipped default with the real BGE encoder."""

    def setUp(self):
        # the default must be tested even when a developer exported COGNIMEM_SCORER (unittest skips tests/__init__.py)
        self.enterContext(patch.dict(os.environ))
        os.environ.pop("COGNIMEM_SCORER", None)

    def test_golden_vectors(self):
        model = load_model()
        scorer = HybridImportanceScorer(model, embedder=default_embedder())
        golden = model.metadata["golden"]
        for vector in golden["vectors"]:
            with self.subTest(text=vector["text"]):
                name, probs = scorer.predict(vector["text"])
                self.assertEqual(name, vector["class"])
                np.testing.assert_allclose(probs, vector["probabilities"], atol=golden["tolerance"])

    def test_default_core_cli_and_properties(self):
        self.assertIsInstance(MemoryCore().importance_scorer, HybridImportanceScorer)
        core = MemoryCore()
        preference = core.process(MemoryInput("I prefer Python examples with clear tests.", "u1", "s1"))
        greeting = core.process(MemoryInput("hello", "u1", "s1"))
        constraint = core.process(MemoryInput("Never include peanuts in my meals.", "u1", "s1"))
        self.assertGreater(preference.importance_score, greeting.importance_score)
        self.assertGreater(constraint.importance_score, greeting.importance_score)
        self.assertEqual(preference.tier.value, "long_term")
        self.assertEqual(greeting.tier.value, "working")
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as output:
            code = cli_main(["--store", str(Path(directory) / "s.jsonl"), "process", "I prefer concise summaries.",
                             "--user-id", "u", "--session-id", "s"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["source_metadata"]["importance_model"]["type"], "hybrid")

    def test_lifecycle_checks_all_pass_with_the_hybrid(self):
        from evaluation import lifecycle
        report = lifecycle.evaluate()
        self.assertEqual(report["importance_model"]["type"], "hybrid")
        self.assertEqual(report["passed"], report["checks"], [r for r in report["results"] if not r["passed"]])


@NEEDS_MODEL
class ApiHybridTests(unittest.TestCase):
    def test_ready_reports_the_hybrid_scorer(self):
        from fastapi.testclient import TestClient
        from api.app import create_app
        from api.config import Settings
        scorer = fake_scorer()
        with tempfile.TemporaryDirectory() as directory:
            settings = dataclasses.replace(Settings.for_testing(Path(directory)), scorer="hybrid")
            # the service builds one encoder for retrieval and scoring; a fake keeps the test free of the real BGE model
            with patch("api.services.local_embedder", lambda threads=None: scorer.embedder):
                app = create_app(settings)
            self.assertIs(app.state.service.rag.embedder, app.state.service.core.importance_scorer.embedder)
            with TestClient(app) as client:
                body = client.get("/health/ready").json()["data"]
        self.assertEqual(body["importance_scorer"], {"type": "hybrid", "model_sha256": load_model().sha256})


if __name__ == "__main__":
    unittest.main()
