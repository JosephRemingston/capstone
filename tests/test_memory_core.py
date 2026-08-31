from __future__ import annotations

import unittest
import sys
from datetime import datetime, timedelta, timezone

from main import (
    FeatureExtractor,
    ImportanceScorer,
    LifecycleManager,
    MemoryCategory,
    MemoryClassifier,
    MemoryCore,
    MemoryInput,
    MemoryRecord,
    MemoryTier,
)


class MemoryClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = MemoryClassifier()

    def classify(self, content: str) -> MemoryCategory:
        return self.classifier.classify(MemoryInput(content=content, user_id="u1", session_id="s1"))

    def test_classifies_memory_types(self) -> None:
        cases = {
            "My company is Acme Robotics.": MemoryCategory.SEMANTIC,
            "Yesterday I met Prof. Karnam Balaji.": MemoryCategory.EPISODIC,
            "How to deploy: first run tests, then build Docker.": MemoryCategory.PROCEDURAL,
            "I prefer concise technical summaries.": MemoryCategory.PREFERENCE,
            "Remind me to review the capstone report tomorrow.": MemoryCategory.TASK,
            "Thanks": MemoryCategory.TEMPORARY,
        }

        for content, expected in cases.items():
            with self.subTest(content=content):
                self.assertEqual(self.classify(content), expected)


class ScoringAndLifecycleTests(unittest.TestCase):
    def test_durable_memory_scores_higher_than_greeting(self) -> None:
        core = MemoryCore()
        preference = core.process(
            MemoryInput(
                content="I prefer Python examples with clear tests.",
                user_id="u1",
                session_id="s1",
            )
        )
        greeting = core.process(MemoryInput(content="hello", user_id="u1", session_id="s1"))

        self.assertGreater(preference.importance_score, greeting.importance_score)
        self.assertEqual(preference.tier, MemoryTier.LONG_TERM)
        self.assertEqual(greeting.tier, MemoryTier.WORKING)

    def test_lifecycle_assigns_expected_tiers(self) -> None:
        manager = LifecycleManager()
        now = datetime.now(timezone.utc)

        temporary = MemoryRecord(
            content="ok",
            user_id="u1",
            session_id="s1",
            category=MemoryCategory.TEMPORARY,
            created_at=now,
            updated_at=now,
        )
        episodic = MemoryRecord(
            content="Yesterday I submitted the proposal.",
            user_id="u1",
            session_id="s1",
            category=MemoryCategory.EPISODIC,
            created_at=now,
            updated_at=now,
        )
        semantic = MemoryRecord(
            content="My guide is Prof. Karnam Balaji.",
            user_id="u1",
            session_id="s1",
            category=MemoryCategory.SEMANTIC,
            created_at=now,
            updated_at=now,
        )
        stale = MemoryRecord(
            content="I used an older memory often.",
            user_id="u1",
            session_id="s1",
            category=MemoryCategory.SEMANTIC,
            created_at=now - timedelta(days=120),
            updated_at=now,
            access_count=2,
        )

        self.assertEqual(manager.assign_tier(temporary, 0.1), MemoryTier.WORKING)
        self.assertEqual(manager.assign_tier(episodic, 0.45), MemoryTier.SHORT_TERM)
        self.assertEqual(manager.assign_tier(semantic, 0.8), MemoryTier.LONG_TERM)
        self.assertEqual(manager.assign_tier(stale, 0.8), MemoryTier.ARCHIVE)


class SerializationTests(unittest.TestCase):
    def test_memory_record_serializes_stably(self) -> None:
        core = MemoryCore()
        record = core.process(
            MemoryInput(
                content="I like retrieval systems that combine ranking signals.",
                user_id="u1",
                session_id="s1",
                metadata={"interaction_score": 0.5, "source": "unit-test"},
            )
        )

        payload = record.to_dict()
        restored = MemoryRecord.from_dict(payload)

        self.assertEqual(restored.to_dict(), payload)
        self.assertEqual(restored.category, MemoryCategory.PREFERENCE)
        self.assertEqual(restored.tier, MemoryTier.LONG_TERM)


class GraphIsolationTests(unittest.TestCase):
    def test_memory_core_does_not_require_graph_modules(self) -> None:
        core = MemoryCore()
        record = core.process(MemoryInput(content="My project title is CogniMem.", user_id="u1", session_id="s1"))

        self.assertEqual(record.category, MemoryCategory.SEMANTIC)
        self.assertNotIn("neo4j", FeatureExtractor.__module__)
        self.assertNotIn("neo4j", ImportanceScorer.__module__)
        self.assertNotIn("neo4j", LifecycleManager.__module__)
        self.assertNotIn("neo4j", sys.modules)
        self.assertNotIn("py2neo", sys.modules)


if __name__ == "__main__":
    unittest.main()
