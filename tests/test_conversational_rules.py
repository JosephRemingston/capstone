"""Developer-authored regression cases, not an independently labeled benchmark."""
import unittest
from datetime import datetime, timezone

from main import MemoryCore, MemoryInput
from main.classification import MemoryClassifier
from main.features import FeatureExtractor
from main.ml import MLFeatureExtractor


class ConversationalRuleTests(unittest.TestCase):
    def process(self, text):
        return MemoryCore().process(MemoryInput(content=text, user_id="u", session_id="s"))

    def test_categories_and_tiers(self):
        cases = [
            ("My favorite book is Dune.", "preference", "long_term"),
            ("Never include peanuts in my meals.", "preference", "long_term"),
            ("I'm allergic to peanuts.", "preference", "long_term"),
            ("I’m allergic to shellfish.", "preference", "long_term"),
            ("Always use metric units in my recipes.", "preference", "long_term"),
            ("The meeting has been cancelled.", "episodic", "short_term"),
            ("The appointment was postponed.", "episodic", "short_term"),
            ("I finished the task today.", "episodic", "short_term"),
            ("I have already submitted my report today.", "episodic", "short_term"),
            ("Yesterday I met my project guide.", "episodic", "short_term"),
            ("I submitted my report today.", "episodic", "short_term"),
            ("Buy groceries today.", "task", "short_term"),
            ("Book a flight tomorrow.", "task", "short_term"),
            ("Book my favorite restaurant tomorrow.", "task", "short_term"),
            ("Schedule a meeting tomorrow.", "task", "short_term"),
            ("Buy milk.", "task", "short_term"),
            ("Remind me to take a walk every day.", "task", "long_term"),
            ("Remind me to submit the report tomorrow.", "task", "short_term"),
            ("My hobby is scrapbooking.", "semantic", "long_term"),
            ("My name is Matthew.", "semantic", "long_term"),
            ("First run tests, then deploy the application.", "procedural", "long_term"),
            ("Hello!", "temporary", "working"),
            ("Thank you!", "temporary", "working"),
        ]
        for content, category, tier in cases:
            with self.subTest(content=content):
                record = self.process(content)
                self.assertEqual(record.category.value, category)
                self.assertEqual(record.tier.value, tier)

    def test_deadlines_are_not_just_mentions_of_today(self):
        for text in ("I submitted my report today.", "Yesterday I met my guide.", "The meeting has been cancelled."):
            self.assertEqual(self.process(text).features["has_deadline"], 0)
        for text in ("Submit the report tomorrow.", "I need to buy groceries today."):
            self.assertEqual(self.process(text).features["has_deadline"], 1)
        self.assertEqual(self.process("I have not completed the task due tomorrow.").category.value, "task")

    def test_temporary_vs_constraint_order_and_expiry(self):
        greeting = self.process("hello")
        constraint = self.process("Never include peanuts in my meals.")
        task = self.process("Buy groceries today.")
        self.assertGreater(constraint.importance_score, greeting.importance_score)
        self.assertIsNone(constraint.expires_at)
        self.assertIsNotNone(task.expires_at)
        self.assertIsNone(task.archive_after)

    def test_legacy_feature_path_is_preserved(self):
        incoming = MemoryInput(content="I submitted my report today.", user_id="u", session_id="s",
                               timestamp=datetime.now(timezone.utc))
        from main import MemoryRecord
        record = MemoryRecord.from_input(incoming, MemoryClassifier(legacy=True).classify(incoming))
        old = FeatureExtractor(legacy=True).extract(incoming, record)
        ml = MLFeatureExtractor().extract(incoming, record)
        self.assertEqual(old["has_deadline"], 1)
        self.assertEqual(ml["has_deadline"], 1)
        self.assertEqual(MemoryClassifier(legacy=True).classify(MemoryInput(
            content="My favorite book is Dune.", user_id="u", session_id="s")).value, "task")


if __name__ == "__main__":
    unittest.main()
