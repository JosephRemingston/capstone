import copy
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluation.ablation import load_store
from evaluation.build_dataset import build, compile_persona, load_seeds, serialize, validate_content
from evaluation.schema import DATASET_PATH, GOLD_USER, QUERY_TYPES, load_dataset
from main.retrieval.rag import MemoryRAG, graph_predicates

from tests._evaluation_fakes import HashEmbedder


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(DATASET_PATH.read_text())

    def test_saved_file_matches_a_fresh_build(self):
        self.assertEqual(DATASET_PATH.read_text(), serialize(build()))

    def test_counts_split_and_provenance(self):
        queries = self.payload['queries']
        self.assertEqual(len(queries), 360)
        self.assertEqual(set(Counter(q['query_type'] for q in queries).values()), {30})
        self.assertGreaterEqual(len(self.payload['memories']), 440)
        self.assertEqual(self.payload['splits']['dev'], ['p01', 'p02', 'p03'])
        dev = {q['persona_id'] for q in queries if q['split'] == 'dev'}
        test = {q['persona_id'] for q in queries if q['split'] == 'test'}
        self.assertFalse(dev & test)
        self.assertIn('not human-validated', self.payload['label_provenance'])
        for split in ('dev', 'test'):
            _, selected = load_dataset(DATASET_PATH, split)
            self.assertEqual({q['query_type'] for q in selected}, set(QUERY_TYPES))

    def test_raw_loading_keeps_every_gold_memory_visible(self):
        with TemporaryDirectory() as directory:
            store = load_store(self.payload['memories'], Path(directory) / 'store.jsonl', 'save')
            self.assertEqual(len(store.list(user_id=GOLD_USER)), len(self.payload['memories']))

    def test_every_multi_hop_query_has_a_cued_graph_path(self):
        with TemporaryDirectory() as directory:
            store = load_store(self.payload['memories'], Path(directory) / 'store.jsonl', 'save')
            rag = MemoryRAG(store, embedder=HashEmbedder(), reranker=None, ann_enabled=False)
            rag.sync(user_id=GOLD_USER, semantic=False)
            for query in (q for q in self.payload['queries'] if q['query_type'] == 'multi_hop'):
                seeds = [s['id'] for s in rag.graph.entities(query['query'], user_id=GOLD_USER)]
                paths = rag.graph.traverse(seeds, user_id=GOLD_USER, max_hops=2)
                cued = graph_predicates(query['query'])
                self.assertTrue(any(set(query['supporting_memory_ids']) <= set(p['source_ids'])
                                    and all(e['predicate'] in cued for e in p['edges']) for p in paths),
                                query['query_id'])

    def test_validators_reject_unattainable_answers_and_bad_history(self):
        seeds = load_seeds()
        broken = copy.deepcopy(seeds)
        target = broken[0]
        queries = target['queries']
        queries[0]['answer'] = 'astrophysics'
        historical = next(q for q in queries if q['type'] == 'historical')
        historical['as_of'] = '2030-01-01'
        memories, compiled = [], []
        for seed in broken:
            m, q = compile_persona(seed)
            memories += m
            compiled += q
        payload = {**self.payload, 'memories': memories, 'queries': compiled}
        with self.assertRaises(ValueError) as error:
            validate_content(payload, broken)
        self.assertIn('no answer form appears in a supporting memory', str(error.exception))
        self.assertIn('no supporting relation is valid at as_of', str(error.exception))


if __name__ == '__main__':
    unittest.main()
