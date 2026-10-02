from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from main.application.core import MemoryCore
from main.domain.models import MemoryInput
from main.retrieval.rag import MemoryRAG
from main.retrieval.embeddings import normalized
from main.storage.indexes import ApproximateVectorIndex, IndexDatabase
from main.storage.store import LocalMemoryStore


class TinyEmbedder:
    model_id = "test-ann-v1"
    dimension = 4

    def passages(self, texts):
        return [self.query(text) for text in texts]

    def query(self, text):
        t = text.lower()
        values = [1.0 if "chennai" in t else 0.0,
                  1.0 if "bengaluru" in t else 0.0,
                  1.0 if "python" in t else 0.0,
                  1.0 if "database" in t else 0.0]
        if not any(values):
            values[3] = 0.5
        return normalized(values, self.dimension)


class ANNTests(unittest.TestCase):
    def test_approximate_index_builds_and_searches_buckets(self):
        with TemporaryDirectory() as directory:
            store = LocalMemoryStore(Path(directory) / "memories.jsonl")
            core = MemoryCore()
            records = []
            for text in ["I live in Chennai", "I live in Bengaluru", "I use Python", "I work with a database"]:
                records.append(store.ingest(core.process(MemoryInput(text, "u", "s"))))
            db = IndexDatabase(Path(directory) / "index.sqlite3")
            index = ApproximateVectorIndex(db, TinyEmbedder(), tables=6, bits=8)
            result = index.sync(records, user_id="u")
            self.assertEqual(result["ann"]["algorithm"], "random_hyperplane_lsh")
            hits = index.search("Chennai", user_id="u", allowed_ids={r.id for r in records}, limit=5, minimum=0.0)
            self.assertTrue(hits)
            self.assertEqual(hits[0]["memory_id"], records[0].id)

    def test_rag_switches_to_ann_after_threshold(self):
        with TemporaryDirectory() as directory:
            store = LocalMemoryStore(Path(directory) / "memories.jsonl")
            core = MemoryCore()
            for i in range(3):
                store.ingest(core.process(MemoryInput(f"Python note {i}", "u", "s")))
            rag = MemoryRAG(store, embedder=TinyEmbedder(), index_path=Path(directory) / "index.sqlite3",
                            reranker=False, ann_enabled=True, ann_exact_threshold=2)
            self.assertIsInstance(rag.vectors, ApproximateVectorIndex)
