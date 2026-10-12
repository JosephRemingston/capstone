from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import random
import struct
from tempfile import TemporaryDirectory
import unittest

from main.application.core import MemoryCore
from main.domain.models import MemoryCategory, MemoryInput, MemoryRecord, MemoryTier
from main.retrieval.rag import MemoryRAG
from main.retrieval.embeddings import normalized
from main.storage.indexes import ApproximateVectorIndex, IndexDatabase, VectorIndex
from main.storage.store import LocalMemoryStore

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


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


class ClusteredEmbedder:
    """Seeded 32-d vectors: 'topic T ...' -> unit centre T plus Gaussian noise (item-item cosine ~0.5)."""
    model_id = "test-clustered-32"
    dimension = 32

    def __init__(self, topics=40, noise=1.0, seed=7):
        rng = random.Random(seed)
        self.centers = [normalized([rng.gauss(0, 1) for _ in range(self.dimension)], self.dimension)
                        for _ in range(topics)]
        self.scale = noise / self.dimension ** 0.5

    def query(self, text):
        topic = int(text.split()[1])
        rng = random.Random(sha256(text.encode()).hexdigest())
        return normalized([c + rng.gauss(0, self.scale) for c in self.centers[topic]], self.dimension)

    def passages(self, texts):
        return [self.query(text.split("\n")[-1]) for text in texts]


def synthetic_records(count, topics=40):
    return [MemoryRecord(id=f"m{i:04d}", content=f"topic {i % topics} item {i}", user_id="u", session_id="s",
                         category=MemoryCategory.SEMANTIC, tier=MemoryTier.LONG_TERM,
                         created_at=NOW, updated_at=NOW, recorded_at=NOW) for i in range(count)]


def legacy_exact(index, query, *, user_id, allowed_ids, limit, minimum):
    """The pre-numpy VectorIndex.search loop, kept as the ordering oracle."""
    vector = normalized(index.embedder.query(query), index.embedder.dimension)
    results = []
    with index.database.connect() as db:
        for row in db.execute("SELECT * FROM vectors WHERE user_id=? AND model=?", (user_id, index.embedder.model_id)):
            if row["memory_id"] in allowed_ids:
                stored = struct.unpack("<" + "f" * row["dimension"], row["vector"])
                score = sum(a * b for a, b in zip(vector, stored))
                if score >= minimum:
                    results.append((row["chunk_id"], score))
    return sorted(results, key=lambda item: (-item[1], item[0]))[:limit]


class ANNTests(unittest.TestCase):
    def test_approximate_index_builds_and_searches_buckets(self):
        with TemporaryDirectory() as directory:
            store = LocalMemoryStore(Path(directory) / "memories.jsonl")
            core = MemoryCore()
            records = []
            for text in ["I live in Chennai", "I live in Bengaluru", "I use Python", "I work with a database"]:
                records.append(store.ingest(core.process(MemoryInput(text, "u", "s"))))
            db = IndexDatabase(Path(directory) / "index.sqlite3")
            # min_candidates=1 keeps this tiny corpus on the LSH path instead of the exact shortcut.
            index = ApproximateVectorIndex(db, TinyEmbedder(), tables=6, bits=8, min_candidates=1)
            result = index.sync(records, user_id="u")
            self.assertEqual(result["ann"]["algorithm"], "random_hyperplane_lsh")
            hits = index.search("Chennai", user_id="u", allowed_ids={r.id for r in records}, limit=1, minimum=0.0)
            self.assertEqual(index.last_search["algorithm"], "lsh")
            self.assertEqual(hits[0]["memory_id"], records[0].id)

    def test_lsh_recall_guard(self):
        """2,000 clustered chunks, 50 queries: LSH top-20 must match the exact top-20 >= 95%."""
        with TemporaryDirectory() as directory:
            records = synthetic_records(2000)
            allowed = {r.id for r in records}
            db = IndexDatabase(Path(directory) / "index.sqlite3")
            ann = ApproximateVectorIndex(db, ClusteredEmbedder())
            ann.sync(records, user_id="u")
            exact = VectorIndex(db, ann.embedder)
            overlaps, fallbacks, scanned = [], 0, []
            for q in range(50):
                query = f"topic {q % 40} query {q}"
                truth = {h["chunk_id"] for h in exact.search(query, user_id="u", allowed_ids=allowed, limit=20, minimum=0.0)}
                found = {h["chunk_id"] for h in ann.search(query, user_id="u", allowed_ids=allowed, limit=20, minimum=0.0)}
                overlaps.append(len(truth & found) / len(truth))
                fallbacks += ann.last_search["fallback"]
                scanned.append(ann.last_search["candidates"])
            self.assertGreaterEqual(sum(overlaps) / len(overlaps), 0.95)
            self.assertLessEqual(fallbacks / 50, 0.1, "fallback rate")
            # The guard is only meaningful if LSH actually prunes the corpus.
            self.assertLess(sum(scanned) / len(scanned), 0.5 * len(records))

    def test_sparse_buckets_fall_back_to_exact_search(self):
        with TemporaryDirectory() as directory:
            records = synthetic_records(300)
            allowed = {r.id for r in records}
            db = IndexDatabase(Path(directory) / "index.sqlite3")
            ann = ApproximateVectorIndex(db, ClusteredEmbedder(), min_candidates=290)
            ann.sync(records, user_id="u")
            hits = ann.search("topic 3 query 1", user_id="u", allowed_ids=allowed, limit=10, minimum=0.0)
            self.assertTrue(ann.last_search["fallback"])
            self.assertEqual(ann.last_search["algorithm"], "exact_fallback")
            expected = VectorIndex(db, ann.embedder).search("topic 3 query 1", user_id="u", allowed_ids=allowed,
                                                            limit=10, minimum=0.0)
            self.assertEqual([h["chunk_id"] for h in hits], [h["chunk_id"] for h in expected])

    def test_fingerprint_covers_lsh_parameters_and_rebuilds(self):
        with TemporaryDirectory() as directory:
            records = synthetic_records(50)
            db = IndexDatabase(Path(directory) / "index.sqlite3")
            embedder = ClusteredEmbedder()
            rows = [{"chunk_id": "a", "digest": "x"}]
            prints = {ApproximateVectorIndex(db, embedder, tables=t, bits=b, seed=s)._fingerprint(rows)
                      for t, b, s in ((8, 10, 1729), (6, 10, 1729), (8, 12, 1729), (8, 10, 7))}
            self.assertEqual(len(prints), 4)
            ApproximateVectorIndex(db, embedder, tables=6).sync(records, user_id="u")
            ApproximateVectorIndex(db, embedder, tables=8).sync(records, user_id="u")
            with db.connect() as connection:
                count = connection.execute("SELECT COUNT(*) FROM lsh_buckets WHERE user_id='u'").fetchone()[0]
            self.assertEqual(count, 8 * len(records))

    def test_numpy_exact_scan_preserves_legacy_ordering(self):
        with TemporaryDirectory() as directory:
            records = synthetic_records(600)
            allowed = {r.id for r in records[::2]}
            index = VectorIndex(IndexDatabase(Path(directory) / "index.sqlite3"), ClusteredEmbedder())
            index.sync(records, user_id="u")
            for q in range(10):
                query = f"topic {q} query {q}"
                for limit, minimum in ((10 ** 6, -1.0), (60, 0.35)):
                    new = index.search(query, user_id="u", allowed_ids=allowed, limit=limit, minimum=minimum)
                    old = legacy_exact(index, query, user_id="u", allowed_ids=allowed, limit=limit, minimum=minimum)
                    self.assertEqual([h["chunk_id"] for h in new], [chunk for chunk, _ in old])
                    for hit, (_, score) in zip(new, old):
                        self.assertAlmostEqual(hit["score"], score, places=12)
                        self.assertIsInstance(hit["score"], float)

    def test_rag_defaults_to_cached_exact_index_without_listing_store(self):
        with TemporaryDirectory() as directory:
            class NoListStore(LocalMemoryStore):
                def list(self, *args, **kwargs):
                    raise AssertionError("store.list() must not run when ANN is disabled")

            store = NoListStore(Path(directory) / "memories.jsonl")
            rag = MemoryRAG(store, embedder=TinyEmbedder(), index_path=Path(directory) / "index.sqlite3", reranker=None)
            self.assertFalse(rag.ann_enabled)
            self.assertEqual(rag.ann_exact_threshold, 5000)
            first = rag.vectors
            self.assertIsInstance(first, VectorIndex)
            self.assertIs(first, rag.vectors)

    def test_rag_switches_to_ann_after_threshold(self):
        with TemporaryDirectory() as directory:
            store = LocalMemoryStore(Path(directory) / "memories.jsonl")
            core = MemoryCore()
            for i in range(3):
                store.ingest(core.process(MemoryInput(f"Python note {i}", "u", "s")))
            rag = MemoryRAG(store, embedder=TinyEmbedder(), index_path=Path(directory) / "index.sqlite3",
                            reranker=False, ann_enabled=True, ann_exact_threshold=2)
            self.assertIsInstance(rag.vectors, ApproximateVectorIndex)
            self.assertIs(rag.vectors, rag.vectors)
            hits = rag.search("python", user_id="u", mode="semantic", limit=2)
            self.assertEqual(hits[0]["vector_index"]["algorithm"], "exact")  # 3 memories < min_candidates

    def test_invalid_ann_options_are_rejected(self):
        with TemporaryDirectory() as directory:
            store = LocalMemoryStore(Path(directory) / "memories.jsonl")
            for options in ({"ann_exact_threshold": 0}, {"ann_enabled": "yes"}, {"candidate_pool": 201},
                            {"candidate_pool": 0}, {"model_threads": 0}, {"model_threads": True}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    MemoryRAG(store, embedder=TinyEmbedder(), **options)
            with self.assertRaises(ValueError):
                ApproximateVectorIndex(IndexDatabase(Path(directory) / "i.sqlite3"), TinyEmbedder(), min_candidates=0)


if __name__ == "__main__":
    unittest.main()
