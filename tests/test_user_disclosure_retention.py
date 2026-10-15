"""Developer-authored regression cases for keeping what users say about themselves."""
import unittest
from datetime import datetime, timedelta, timezone

from evaluation.clock import pinned_clock
from main import MemoryCore, MemoryInput
from main.domain.classification import MemoryClassifier
from main.domain.text_rules import phatic, self_disclosure

NOW = datetime(2026, 10, 12, 9, tzinfo=timezone.utc)


def process(text, role='user'):
    return MemoryCore.with_heuristic().process(MemoryInput(text, 'u', 's', role=role, timestamp=NOW))


class UserDisclosureRetentionTests(unittest.TestCase):
    def setUp(self):
        # The heuristic's recency feature reads the clock; score "at" NOW, as production scores a new memory.
        self.enterContext(pinned_clock(NOW))

    def test_first_person_content_inside_questions_and_hedges_is_kept(self):
        for text in ["I'm planning a trip to Lisbon in May, can you suggest neighbourhoods?",
                     'I have a golden retriever who hates baths, any tips?',
                     'Do you know if I need a visa to visit Japan?',
                     'What are some ways to keep my living room dust-free?',
                     "I'm not sure if my mom would like a Kindle for her birthday.",
                     'I was thinking about selling my guitar collection, maybe online?']:
            with self.subTest(text=text):
                record = process(text)
                self.assertEqual(record.category.value, 'episodic')
                self.assertEqual(record.tier.value, 'short_term')
                self.assertEqual(record.expires_at, NOW + timedelta(days=90))
                self.assertIsNone(record.claim)  # an observation, never an asserted fact

    def test_plain_first_person_statements_reach_episodic_instead_of_temporary(self):
        for text in ["I'm vegan.", "I'd like to set up a budget because I'm saving for a house."]:
            with self.subTest(text=text):
                self.assertEqual(process(text).category.value, 'episodic')

    def test_phatic_and_generic_messages_stay_working_memory(self):
        for text in ['Thanks!', 'ok', 'lol', 'Never mind.', 'Can you say that again?', 'I see.',
                     'I agree', 'Got it, thanks.', "I didn't catch that, can you repeat it?",
                     'I have no idea', 'Hey, good morning!', 'What is the capital of Australia?',
                     'Can you tell me more about rate locks?', 'Can I also get a room with a fireplace?',
                     'I think that is a great idea!', 'Can you summarize the causes of World War I?']:
            with self.subTest(text=text):
                record = process(text)
                self.assertEqual(record.category.value, 'temporary')
                self.assertEqual(record.tier.value, 'working')

    def test_reported_speech_and_assistant_first_person_are_not_user_disclosures(self):
        self.assertFalse(self_disclosure('She said "I moved to Berlin".'))
        self.assertEqual(process('She said "I moved to Berlin".').category.value, 'temporary')
        reply = "I'd be happy to help! I recommend visiting Alfama, what dates are you considering?"
        self.assertEqual(process(reply, role='assistant').category.value, 'temporary')
        self.assertEqual(process(reply, role='assistant').tier.value, 'working')

    def test_user_events_get_ninety_days_and_other_short_term_memories_keep_two_weeks(self):
        # Scope: every user-role EPISODIC short-term memory (disclosures, reported events and
        # task updates) is kept for archive_age_days; tasks and assistant turns keep 14 days.
        self.assertEqual(process('Buy groceries').expires_at, NOW + timedelta(days=14))
        self.assertEqual(process('I submitted my report today.').expires_at, NOW + timedelta(days=90))
        episode = process('I submitted my report today.', role='assistant')
        self.assertEqual(episode.expires_at, NOW + timedelta(days=14))

    def test_explicit_relations_still_become_long_term_semantic_facts(self):
        record = MemoryCore.with_heuristic().process(MemoryInput(
            "I'm at Acme now, can you update my profile?", 'u', 's', timestamp=NOW,
            metadata={'entities': [{'id': 'org:acme', 'type': 'organization', 'label': 'Acme'}],
                      'relations': [{'subject': 'self', 'predicate': 'works_at', 'object': 'org:acme'}]}))
        self.assertEqual(record.category.value, 'semantic')
        self.assertEqual(record.tier.value, 'long_term')
        self.assertIsNone(record.expires_at)

    def test_roman_numeral_i_is_not_the_pronoun(self):
        self.assertFalse(self_disclosure('Tell me about Elizabeth I.'))
        self.assertFalse(self_disclosure('What is a type I error in statistics?'))
        self.assertTrue(self_disclosure('Yesterday I went to the dentist, is that normal?'))

    def test_legacy_classifier_ignores_disclosure_rules(self):
        incoming = MemoryInput('Do you know if I need a permit for the trail?', 'u', 's')
        self.assertEqual(MemoryClassifier(legacy=True).classify(incoming).value, 'temporary')
        self.assertEqual(MemoryClassifier().classify(incoming).value, 'episodic')
        self.assertTrue(phatic('Thanks, that makes sense.'))
        self.assertFalse(phatic('Thanks, I moved to Pune last week.'))



class ReviewFixTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(pinned_clock(NOW))

    def test_recall_questions_are_not_new_disclosures(self):
        for text in ["What was my dog's name again?", 'Wait, what is my dog called?']:
            with self.subTest(text=text):
                self.assertFalse(self_disclosure(text))
                self.assertEqual(process(text).category.value, 'temporary')
        self.assertTrue(self_disclosure("What's my best option given I'm moving to Oslo next month?"))

    def test_date_word_requests_keep_the_two_week_window(self):
        record = process('Explain quantum computing today')
        if record.category.value == 'episodic':
            self.assertEqual(record.expires_at, NOW + timedelta(days=14))


if __name__ == '__main__':
    unittest.main()
