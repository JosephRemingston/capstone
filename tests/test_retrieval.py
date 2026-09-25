from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from main import LocalMemoryStore, MemoryCategory, MemoryRanker, MemoryRecord, MemoryTier
from main.__main__ import main as cli_main


class RankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 9, 25, tzinfo=timezone.utc)
        self.record = MemoryRecord(
            id="base", content="Python testing", user_id="u1", session_id="s1",
            category=MemoryCategory.SEMANTIC, tier=MemoryTier.LONG_TERM,
            created_at=self.now, updated_at=self.now, importance_score=0.5,
        )
        self.ranker = MemoryRanker()

    def test_each_signal_changes_order_independently(self) -> None:
        worse_records = {
            "keyword": replace(self.record, content="Python"),
            "category": replace(self.record, category=MemoryCategory.TEMPORARY),
            "tier": replace(self.record, tier=MemoryTier.ARCHIVE),
            "recency": replace(self.record, created_at=self.now - timedelta(days=30)),
            "importance": replace(self.record, importance_score=0.1),
        }
        for signal, worse in worse_records.items():
            with self.subTest(signal=signal):
                worse = replace(worse, id="a")
                better = replace(self.record, id="z")
                self.assertEqual(
                    self.ranker.rank("Python testing", [worse, better], now=self.now),
                    [better, worse],
                )

    def test_keyword_coverage_beats_repeated_mentions(self) -> None:
        repeated = replace(self.record, id="repeat", content="Python " * 100)
        self.assertEqual(
            self.ranker.rank("Python testing", [repeated, self.record], now=self.now),
            [self.record, repeated],
        )

    def test_case_punctuation_duplicates_and_whole_words(self) -> None:
        substring = replace(self.record, id="substring", content="Pythonic testers")
        expected = [self.record]
        for query in ("PYTHON, testing!", "python python testing", "python-testing"):
            with self.subTest(query=query):
                self.assertEqual(self.ranker.rank(query, [substring, self.record], now=self.now), expected)
        for query in ("", "   ", "!!!", "unrelated"):
            self.assertEqual(self.ranker.rank(query, [self.record], now=self.now), [])

    def test_unrelated_high_value_memory_is_excluded(self) -> None:
        unrelated = replace(self.record, content="Database administration", importance_score=1)
        self.assertEqual(self.ranker.rank("python", [unrelated, self.record], now=self.now), [self.record])

    def test_ties_are_deterministic_and_future_recency_is_capped(self) -> None:
        a = replace(self.record, id="a")
        b = replace(self.record, id="b")
        self.assertEqual(self.ranker.rank("python", [b, a], now=self.now), [a, b])
        future = replace(self.record, id="future", created_at=self.now + timedelta(days=30))
        higher_importance = replace(self.record, importance_score=0.6)
        self.assertEqual(
            self.ranker.rank("python", [future, higher_importance], now=self.now),
            [higher_importance, future],
        )

    def test_custom_weights_can_change_ranking(self) -> None:
        important = replace(self.record, id="important", content="Python", importance_score=1)
        self.assertEqual(self.ranker.rank("python testing", [important, self.record], now=self.now)[0], self.record)
        weights = dict.fromkeys(self.ranker.weights, 0.0)
        weights["importance"] = 1.0
        self.assertEqual(MemoryRanker(weights=weights).rank("python testing", [self.record, important], now=self.now)[0], important)

    def test_invalid_configuration_is_rejected(self) -> None:
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MemoryRanker(recency_half_life_days=value)
        for weights in ({}, dict.fromkeys(self.ranker.weights, 0), {**self.ranker.weights, "recency": -1}):
            with self.assertRaises(ValueError):
                MemoryRanker(weights=weights)

    def test_store_filters_limits_cli_and_read_only_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memories.jsonl"
            store = LocalMemoryStore(path)
            lower = replace(self.record, id="lower", importance_score=0)
            other_user = replace(self.record, id="other-user", user_id="u2", importance_score=1)
            other_session = replace(self.record, id="other-session", session_id="s2")
            other_category = replace(self.record, id="other-category", category=MemoryCategory.TASK)
            other_tier = replace(self.record, id="other-tier", tier=MemoryTier.WORKING)
            for record in (lower, other_user, other_session, other_category, other_tier, self.record):
                store.save(record)
            original = path.read_bytes()
            filters = dict(user_id="u1", session_id="s1", category="semantic", tier="long_term")
            self.assertEqual(store.search("python", **filters), [self.record, lower])
            self.assertEqual(store.search("python", **filters, limit=1), [self.record])
            self.assertEqual(store.search("python", limit=0), [])
            with self.assertRaises(ValueError):
                store.search("python", limit=-1)
            output = io.StringIO()
            with redirect_stdout(output):
                status = cli_main([
                    "--store", str(path), "search", "python", "--user-id", "u1",
                    "--session-id", "s1", "--category", "semantic", "--tier", "long_term",
                ])
            self.assertEqual(status, 0)
            self.assertEqual([item["id"] for item in json.loads(output.getvalue())], ["base", "lower"])
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
