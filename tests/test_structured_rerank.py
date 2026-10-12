"""Post-rerank structural ordering: graph evidence, validity, statement versions, evidence chains."""
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from main.domain.models import MemoryRecord
from main.retrieval.rag import ALL_STRUCTURE_STEPS, MemoryRAG, time_intent
from main.storage.store import LocalMemoryStore


class Embedder:
    """Everything is similar to the query except texts naming Porto (orthogonal, below the 0.35 floor)."""
    model_id = 'structure-test-embedder'
    dimension = 2

    def passages(self, texts):
        return [self.query(text) for text in texts]

    def query(self, text):
        return [0.0, 1.0] if 'Porto' in text else [1.0, 0.0]


class BrokenReranker:
    model_id = 'broken-reranker'

    def rerank(self, query, documents):
        raise RuntimeError('model unavailable')


class ScriptedReranker:
    """Scores a document by the first scripted phrase it contains; unknown documents score -5."""
    model_id = 'scripted-reranker'

    def __init__(self, script):
        self.script = script

    def rerank(self, query, documents):
        return [next((score for phrase, score in self.script.items() if phrase in text), -5.0)
                for text in documents]


def record(identifier, content, created, *, speaker=None, entities=(), relations=()):
    metadata = {'entities': list(entities), 'relations': list(relations)}
    if speaker:
        metadata['speaker'] = speaker
    when = created.isoformat()
    return MemoryRecord.from_dict({
        'id': identifier, 'content': content, 'user_id': 'u', 'session_id': 's', 'role': 'user',
        'category': 'semantic', 'tier': 'long_term', 'created_at': when, 'updated_at': when,
        'recorded_at': when, 'confidence': 1.0, 'importance_score': 0.8, 'source_metadata': metadata})


def day(year, month, number=1):
    return datetime(year, month, number, tzinfo=timezone.utc)


LISBON = {'id': 'loc:lisbon', 'type': 'location', 'label': 'Lisbon'}
PORTO = {'id': 'loc:porto', 'type': 'location', 'label': 'Porto'}


class StructuredOrderingTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = LocalMemoryStore(Path(temporary.name) / 'memories.jsonl')

    def rag(self, records, script):
        self.store._append(records)
        reranker = BrokenReranker() if script is None else ScriptedReranker(script)
        return MemoryRAG(self.store, embedder=Embedder(), reranker=reranker, ann_enabled=False)

    def ids(self, hits):
        return [hit['memory_id'] for hit in hits]

    def homes(self):
        return [
            record('old-home', 'I have been renting a flat in Lisbon since 2021', day(2024, 1),
                   entities=[LISBON], relations=[{'subject': 'self', 'predicate': 'lives_in', 'object': 'loc:lisbon',
                                                  'valid_from': day(2021, 4).isoformat(),
                                                  'valid_to': day(2025, 3, 15).isoformat()}]),
            record('new-home', 'We finally moved to Porto last March', day(2025, 6),
                   entities=[PORTO], relations=[{'subject': 'self', 'predicate': 'lives_in', 'object': 'loc:porto',
                                                 'valid_from': day(2025, 3, 15).isoformat()}]),
            record('noise', 'I have a flat white every morning', day(2024, 2)),
        ]

    def test_stale_statement_yields_its_place_to_the_successor(self):
        rag = self.rag(self.homes(), {'Lisbon': 5.0, 'flat white': 4.0, 'Porto': -6.0})
        hits = rag.search('Which city do I live in?', user_id='u', limit=3)
        self.assertEqual(self.ids(hits)[:2], ['new-home', 'old-home'])
        self.assertIn('supersedes:old-home', hits[0]['ordering']['adjustments'])
        self.assertEqual(hits[0]['reranker_score'], -6.0)

    def test_successor_only_replaces_a_stale_statement_the_question_asks_about(self):
        rag = self.rag(self.homes(), {'Lisbon': 5.0, 'flat white': 4.0, 'Porto': -11.0})
        # "How much rent" cues no lives_in relation: the Lisbon memory answers it, Porto must not jump ahead.
        hits = rag.search('How much rent did I pay for my flat in Lisbon?', user_id='u', limit=3)
        self.assertEqual(self.ids(hits)[0], 'old-home')
        self.assertTrue(all(not any(note.startswith('supersedes') for note in hit['ordering']['adjustments'])
                            for hit in hits))

    def test_reranker_failure_injects_no_successor(self):
        rag = self.rag(self.homes(), None)
        hits = rag.search('Which city do I live in?', user_id='u', limit=3)
        self.assertEqual(hits[0]['reranker']['status'], 'fallback')
        self.assertTrue(all(hit['signals'] for hit in hits))

    def test_history_cues_and_disabled_reranking_keep_the_stale_statement(self):
        rag = self.rag(self.homes(), {'Lisbon': 5.0, 'flat white': 4.0, 'Porto': -6.0})
        before = rag.search('Which town was I in before?', user_id='u', limit=3)
        self.assertEqual(self.ids(before)[0], 'old-home')
        self.assertNotIn('new-home', self.ids(before))
        fused = rag.search('Which town am I in?', user_id='u', limit=3, rerank=False)
        self.assertNotIn('new-home', self.ids(fused))
        self.assertTrue(all('ordering' not in hit for hit in fused))

    def test_versions_follow_the_time_intent_of_the_question(self):
        records = [record('retro-old', 'Our team retro happens on Fridays at 4pm', day(2024, 1), speaker='Priya'),
                   record('retro-new', 'Correction: the retro moved to Thursdays at 3', day(2024, 5), speaker='Priya'),
                   record('retro-other', 'Our retro is on Mondays', day(2024, 6), speaker='Jonas')]
        rag = self.rag(records, {'Fridays': 7.0, 'Thursdays': 6.0, 'Mondays': 1.0})
        rag.structure_steps = ALL_STRUCTURE_STEPS   # versions are opt-in
        now = rag.search("Which day is Priya's retro on now?", user_id='u', limit=3)
        self.assertEqual(self.ids(now), ['retro-new', 'retro-old', 'retro-other'])
        self.assertIn('newer_version_of:retro-old', now[0]['ordering']['adjustments'])
        originally = rag.search("Which day was Priya's retro on originally?", user_id='u', limit=3)
        self.assertEqual(self.ids(originally)[:2], ['retro-old', 'retro-new'])
        neutral = rag.search("Which day is Priya's retro on?", user_id='u', limit=3)
        self.assertEqual(self.ids(neutral)[:2], ['retro-old', 'retro-new'])
        rag.structure_steps = MemoryRAG.structure_steps
        self.assertEqual(self.ids(rag.search("Which day is Priya's retro on now?", user_id='u', limit=3))[:2],
                         ['retro-old', 'retro-new'])

    def test_graph_evidence_and_chain_partner_outrank_text_only_distractors(self):
        tidewater = {'id': 'org:tidewater', 'type': 'organization', 'label': 'Tidewater Labs'}
        rotterdam = {'id': 'loc:rotterdam', 'type': 'location', 'label': 'Rotterdam'}
        records = [
            record('job', 'Starting in June I joined Tidewater Labs as a senior analyst', day(2025, 6),
                   entities=[tidewater], relations=[{'subject': 'self', 'predicate': 'works_at',
                                                     'object': 'org:tidewater'}]),
            record('hq', 'Tidewater Labs is headquartered in Rotterdam', day(2025, 6),
                   entities=[tidewater, rotterdam], relations=[{'subject': 'org:tidewater', 'predicate': 'based_in',
                                                                'object': 'loc:rotterdam'}]),
            record('distractor', 'Their old office headquarters were noisy', day(2024, 1)),
        ]
        rag = self.rag(records, {'Rotterdam': 4.0, 'headquarters': 4.5, 'senior analyst': -9.0})
        hits = rag.search('In which city is the company I work for headquartered?', user_id='u', limit=2)
        self.assertEqual(self.ids(hits), ['hq', 'job'])
        self.assertIn('graph_path', hits[0]['ordering']['adjustments'])
        self.assertIn('chain_partner_of:hq', hits[1]['ordering']['adjustments'])
        plain = rag.search('In which city is the company I work for headquartered?', user_id='u', limit=2,
                           channels=('keyword', 'semantic'))
        self.assertEqual(self.ids(plain)[0], 'distractor')
        self.assertTrue(all('ordering' not in hit for hit in plain))

    def test_time_intent(self):
        now = day(2026, 10)
        self.assertEqual(time_intent('Which city does Jonas live in right now?', now), 'current')
        self.assertEqual(time_intent('After the schedule change, which day is the retro?', now), 'current')
        self.assertEqual(time_intent('Which day did Amara first say her lab meeting happened?', now), 'history')
        self.assertEqual(time_intent('Where did Jonas spend Easter in 2024?', now), 'history')
        self.assertEqual(time_intent('What are my plans in 2027?', now), 'neutral')
        self.assertEqual(time_intent('Which language exam did Jonas register to take?', now), 'neutral')
        self.assertEqual(time_intent('What did I originally plan, and what is the plan now?', now), 'neutral')


if __name__ == '__main__':
    unittest.main()
