"""Rebuildable SQLite indexes. JSONL remains the authoritative observation log."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import struct

from ..retrieval.embeddings import Embedder, normalized
from ..domain.models import MemoryRecord

SCHEMA = '''
CREATE TABLE IF NOT EXISTS index_state (kind TEXT, user_id TEXT, fingerprint TEXT, PRIMARY KEY(kind,user_id));
CREATE TABLE IF NOT EXISTS vectors (
 user_id TEXT NOT NULL, model TEXT NOT NULL, chunk_id TEXT NOT NULL, memory_id TEXT NOT NULL,
 text TEXT NOT NULL, start_offset INTEGER, end_offset INTEGER, digest TEXT NOT NULL,
 dimension INTEGER NOT NULL, vector BLOB NOT NULL, PRIMARY KEY(user_id,model,chunk_id));
CREATE INDEX IF NOT EXISTS vector_owner ON vectors(user_id,model,memory_id);
CREATE TABLE IF NOT EXISTS lsh_state (user_id TEXT NOT NULL, model TEXT NOT NULL, fingerprint TEXT NOT NULL, PRIMARY KEY(user_id,model));
CREATE TABLE IF NOT EXISTS lsh_buckets (
 user_id TEXT NOT NULL, model TEXT NOT NULL, table_id INTEGER NOT NULL, bucket INTEGER NOT NULL, chunk_id TEXT NOT NULL,
 PRIMARY KEY(user_id,model,table_id,bucket,chunk_id));
CREATE INDEX IF NOT EXISTS lsh_lookup ON lsh_buckets(user_id,model,table_id,bucket);
CREATE TABLE IF NOT EXISTS nodes (
 user_id TEXT NOT NULL, id TEXT NOT NULL, kind TEXT NOT NULL, label TEXT NOT NULL,
 PRIMARY KEY(user_id,id));
CREATE TABLE IF NOT EXISTS aliases (
 user_id TEXT NOT NULL, alias TEXT NOT NULL, node_id TEXT NOT NULL, known_from TEXT NOT NULL,
 PRIMARY KEY(user_id,alias,node_id));
CREATE INDEX IF NOT EXISTS alias_lookup ON aliases(user_id,alias);
CREATE TABLE IF NOT EXISTS edges (
 user_id TEXT NOT NULL, id TEXT NOT NULL, subject TEXT NOT NULL, predicate TEXT NOT NULL,
 object TEXT NOT NULL, positive INTEGER NOT NULL, source_id TEXT NOT NULL,
 valid_from TEXT NOT NULL, valid_to TEXT, known_from TEXT NOT NULL, known_to TEXT,
 confidence REAL NOT NULL, evidence TEXT NOT NULL, source_revision TEXT, PRIMARY KEY(user_id,id));
CREATE INDEX IF NOT EXISTS edge_subject ON edges(user_id,subject,predicate);
CREATE INDEX IF NOT EXISTS edge_object ON edges(user_id,object,predicate);
CREATE INDEX IF NOT EXISTS edge_time ON edges(user_id,valid_from,valid_to,known_from,known_to);
'''


class IndexDatabase:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(SCHEMA)
            columns = {row['name'] for row in db.execute('PRAGMA table_info(edges)')}
            if 'source_revision' not in columns:
                db.execute('ALTER TABLE edges ADD COLUMN source_revision TEXT')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('PRAGMA foreign_keys=ON')
            with db:
                yield db
        finally:
            db.close()


def digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def document_text(record: MemoryRecord) -> str:
    prefix = []
    for key in ('speaker', 'session_date'):
        value = record.source_metadata.get(key)
        if isinstance(value, str) and value:
            prefix.append(f'{key}: {value}')
    text = '\n'.join([*prefix, record.content])
    if record.summary and record.summary != record.content:
        text += '\nSummary: ' + record.summary
    return text


@dataclass(frozen=True)
class Chunk:
    id: str
    memory_id: str
    text: str
    start: int
    end: int


def chunks(record: MemoryRecord, *, size=1000, overlap=150) -> list[Chunk]:
    if size < 100 or not 0 <= overlap < size:
        raise ValueError('Chunk size must be >=100, with overlap smaller than size')
    text = document_text(record)
    result, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = text.rfind(' ', start + size // 2, end)
            if boundary > start:
                end = boundary
        result.append(Chunk(f'{record.id}:{start}', record.id, text[start:end], start, end))
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return result


class VectorIndex:
    def __init__(self, database: IndexDatabase, embedder: Embedder):
        self.database, self.embedder = database, embedder
        self.last_search: dict = {}

    def sync(self, records: list[MemoryRecord], *, user_id: str) -> dict:
        if not user_id or any(record.user_id != user_id for record in records):
            raise ValueError('Vector indexing requires records from exactly one user')
        desired = {chunk.id: chunk for record in records for chunk in chunks(record)}
        with self.database.connect() as db:
            existing = {row['chunk_id']: row['digest'] for row in db.execute(
                'SELECT chunk_id,digest FROM vectors WHERE user_id=? AND model=?', (user_id, self.embedder.model_id))}
        changed = [chunk for key, chunk in desired.items() if existing.get(key) != digest(chunk.text)]
        vectors = self.embedder.passages([chunk.text for chunk in changed]) if changed else []
        if len(vectors) != len(changed):
            raise ValueError('Embedding provider returned the wrong number of vectors')
        vectors = [normalized(vector, self.embedder.dimension) for vector in vectors]
        # Embedding errors leave the previous index intact; callers never query stale results on failure.
        with self.database.connect() as db:
            for identifier in existing.keys() - desired.keys():
                db.execute('DELETE FROM vectors WHERE user_id=? AND model=? AND chunk_id=?',
                           (user_id, self.embedder.model_id, identifier))
            for chunk, vector in zip(changed, vectors):
                db.execute('INSERT OR REPLACE INTO vectors VALUES (?,?,?,?,?,?,?,?,?,?)',
                           (user_id, self.embedder.model_id, chunk.id, chunk.memory_id, chunk.text,
                            chunk.start, chunk.end, digest(chunk.text), self.embedder.dimension,
                            struct.pack('<' + 'f' * len(vector), *vector)))
        return {'chunks': len(desired), 'embedded': len(changed), 'removed': len(existing.keys() - desired.keys())}

    @staticmethod
    def _validate(query, user_id, limit, minimum) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0 or not -1 <= minimum <= 1 or not user_id:
            raise ValueError('Invalid vector search parameters')

    def _rank(self, vector, rows, *, allowed_ids, limit, minimum) -> list[dict]:
        """Score rows with one float64 matrix product; order is (-score, chunk_id) as before."""
        import numpy as np
        kept = [row for row in rows if row['memory_id'] in allowed_ids]
        if not kept:
            return []
        dimension = self.embedder.dimension
        if any(row['dimension'] != dimension or len(row['vector']) != 4 * dimension for row in kept):
            raise ValueError('Vector index dimension mismatch; rebuild the index')
        matrix = np.frombuffer(b''.join(row['vector'] for row in kept), dtype='<f4').reshape(len(kept), dimension)
        scores = matrix.astype(np.float64) @ np.asarray(vector, dtype=np.float64)
        eligible = np.flatnonzero(scores >= minimum).tolist()
        eligible.sort(key=lambda index: (-scores[index], kept[index]['chunk_id']))
        return [{'memory_id': kept[i]['memory_id'], 'chunk_id': kept[i]['chunk_id'], 'text': kept[i]['text'],
                 'score': float(scores[i]), 'start': kept[i]['start_offset'], 'end': kept[i]['end_offset']}
                for i in eligible[:limit]]

    def search_vector(self, vector, *, user_id: str, allowed_ids: set[str], limit=20, minimum=0.35) -> list[dict]:
        with self.database.connect() as db:
            rows = db.execute('SELECT * FROM vectors WHERE user_id=? AND model=?',
                              (user_id, self.embedder.model_id)).fetchall()
        self.last_search = {'algorithm': 'exact', 'scanned_chunks': len(rows)}
        return self._rank(vector, rows, allowed_ids=allowed_ids, limit=limit, minimum=minimum)

    def search(self, query: str, *, user_id: str, allowed_ids: set[str], limit=20, minimum=0.35) -> list[dict]:
        self._validate(query, user_id, limit, minimum)
        if not query.strip() or not allowed_ids or limit == 0:
            return []
        vector = normalized(self.embedder.query(query), self.embedder.dimension)
        return self.search_vector(vector, user_id=user_id, allowed_ids=allowed_ids, limit=limit, minimum=minimum)


class ApproximateVectorIndex:
    """Random-hyperplane LSH index for approximate cosine search.

    The existing SQLite vector table remains the source of vector data. This index
    adds multiple locality-sensitive hash tables, so large searches inspect only
    vectors that share one or more signatures with the query instead of scanning
    the complete corpus. It is deterministic for a given embedding model.

    Multi-probe widens the Hamming radius (0, 1, 2; within a radius, the buckets
    nearest the query's hyperplane margins first) until at least
    ``max(limit, min_candidates)`` allowed chunks are candidates; when even radius 2
    is too sparse, the search falls back to the exact scan. ``last_search`` records
    which path answered, so recall loss is observable instead of silent.
    """

    VERSION = 2  # bump when bucket assignment changes; it is part of the fingerprint

    def __init__(self, database: IndexDatabase, embedder: Embedder, *, tables: int = 8, bits: int = 10,
                 seed: int = 1729, min_candidates: int = 400):
        if not 1 <= tables <= 32 or not 1 <= bits <= 20:
            raise ValueError('LSH tables must be 1–32 and bits must be 1–20')
        if isinstance(min_candidates, bool) or not isinstance(min_candidates, int) or min_candidates < 1:
            raise ValueError('min_candidates must be a positive integer')
        self.database, self.embedder = database, embedder
        self.tables, self.bits, self.seed, self.min_candidates = tables, bits, seed, min_candidates
        self._planes = self._build_planes()
        self.exact = VectorIndex(database, embedder)
        self.last_search: dict = {}

    def _build_planes(self):
        import random
        import numpy as np
        planes = []
        for table_id in range(self.tables):
            table = []
            for bit in range(self.bits):
                rng = random.Random(f'{self.seed}:{self.embedder.model_id}:{table_id}:{bit}')
                vector = [rng.gauss(0.0, 1.0) for _ in range(self.embedder.dimension)]
                norm = sum(value * value for value in vector) ** 0.5
                table.append([value / norm for value in vector])
            planes.append(table)
        # (tables*bits, dimension); row table_id*bits+bit is the hyperplane for that bit.
        return np.asarray(planes, dtype=np.float64).reshape(self.tables * self.bits, self.embedder.dimension)

    def _fingerprint(self, rows) -> str:
        header = f'v{self.VERSION}:tables={self.tables}:bits={self.bits}:seed={self.seed}:dim={self.embedder.dimension}\n'
        body = ''.join(f"{row['chunk_id']}:{row['digest']}\n" for row in sorted(rows, key=lambda r: r['chunk_id']))
        return sha256((header + body).encode()).hexdigest()

    def _signatures(self, matrix) -> list[list[int]]:
        """One integer bucket per table for every row of ``matrix`` (n, dimension)."""
        import numpy as np
        bits = (np.asarray(matrix, dtype=np.float64) @ self._planes.T) >= 0
        weights = 1 << np.arange(self.bits, dtype=np.int64)
        buckets = bits.reshape(len(bits), self.tables, self.bits).astype(np.int64) @ weights
        return buckets.tolist()

    def _rebuild_buckets(self, user_id: str, rows) -> None:
        import numpy as np
        for row in rows:
            if row['dimension'] != self.embedder.dimension or len(row['vector']) != 4 * self.embedder.dimension:
                raise ValueError('Vector index dimension mismatch; rebuild the index')
        signatures = self._signatures(
            np.frombuffer(b''.join(row['vector'] for row in rows), dtype='<f4').reshape(len(rows), self.embedder.dimension)
        ) if rows else []
        inserts = [(user_id, self.embedder.model_id, table_id, bucket, row['chunk_id'])
                   for row, buckets in zip(rows, signatures) for table_id, bucket in enumerate(buckets)]
        with self.database.connect() as db:
            db.execute('DELETE FROM lsh_buckets WHERE user_id=? AND model=?', (user_id, self.embedder.model_id))
            db.executemany('INSERT INTO lsh_buckets VALUES (?,?,?,?,?)', inserts)

    def sync(self, records: list[MemoryRecord], *, user_id: str) -> dict:
        result = self.exact.sync(records, user_id=user_id)
        with self.database.connect() as db:
            rows = list(db.execute('SELECT chunk_id,digest,dimension,vector FROM vectors WHERE user_id=? AND model=?',
                                   (user_id, self.embedder.model_id)))
            fingerprint = self._fingerprint(rows)
            state = db.execute('SELECT fingerprint FROM lsh_state WHERE user_id=? AND model=?',
                               (user_id, self.embedder.model_id)).fetchone()
        if state is None or state['fingerprint'] != fingerprint:
            self._rebuild_buckets(user_id, rows)
            with self.database.connect() as db:
                db.execute('INSERT OR REPLACE INTO lsh_state VALUES (?,?,?)',
                           (user_id, self.embedder.model_id, fingerprint))
        result['ann'] = {'algorithm': 'random_hyperplane_lsh', 'tables': self.tables, 'bits': self.bits,
                         'min_candidates': self.min_candidates, 'indexed_chunks': len(rows)}
        return result

    def _probes(self, vector) -> list[tuple[float, int, int, int]]:
        """(score, radius, table_id, bucket) for every bucket within Hamming radius 2.

        Query-directed multi-probe: flipping a bit whose projection is close to the
        hyperplane is the likeliest place for a near neighbour to land, so probes are
        ordered by radius and, within a radius, by the summed |projection| of the
        flipped bits.
        """
        import numpy as np
        projection = (self._planes @ np.asarray(vector, dtype=np.float64)).reshape(self.tables, self.bits)
        probes = []
        for table_id, bucket in enumerate(self._signatures([vector])[0]):
            margins = np.abs(projection[table_id]).tolist()
            probes.append((0.0, 0, table_id, bucket))
            for first in range(self.bits):
                probes.append((margins[first], 1, table_id, bucket ^ (1 << first)))
                for second in range(first + 1, self.bits):
                    probes.append((margins[first] + margins[second], 2, table_id,
                                   bucket ^ (1 << first) ^ (1 << second)))
        probes.sort(key=lambda probe: (probe[1], probe[0], probe[2], probe[3]))
        return probes

    def search(self, query: str, *, user_id: str, allowed_ids: set[str], limit=20, minimum=0.35) -> list[dict]:
        VectorIndex._validate(query, user_id, limit, minimum)
        if not query.strip() or not allowed_ids or limit == 0:
            return []
        vector = normalized(self.embedder.query(query), self.embedder.dimension)
        target = max(limit, self.min_candidates)
        if len(allowed_ids) <= target:
            # Too few eligible memories for LSH to save work; scan them exactly.
            self.last_search = {'algorithm': 'exact', 'reason': 'allowed_below_target', 'target': target,
                                'fallback': False}
            return self.exact.search_vector(vector, user_id=user_id, allowed_ids=allowed_ids, limit=limit,
                                            minimum=minimum)
        probes = self._probes(vector)
        seen: set[str] = set()
        rows: list = []
        allowed, used, batch_size = 0, 0, self.tables
        with self.database.connect() as db:
            while used < len(probes):
                batch = probes[used:used + batch_size]
                used += len(batch)
                batch_size = min(batch_size * 2, 256)
                pairs = ','.join('(?,?)' for _ in batch)
                params = [user_id, self.embedder.model_id]
                for _, _, table_id, bucket in batch:
                    params += [table_id, bucket]
                fresh = {row['chunk_id'] for row in db.execute(
                    f'SELECT chunk_id FROM lsh_buckets WHERE user_id=? AND model=? '
                    f'AND (table_id,bucket) IN (VALUES {pairs})', params)} - seen
                seen |= fresh
                fresh_list = sorted(fresh)
                for offset in range(0, len(fresh_list), 500):
                    chunk_ids = fresh_list[offset:offset + 500]
                    marks = ','.join('?' for _ in chunk_ids)
                    for row in db.execute(f'SELECT * FROM vectors WHERE user_id=? AND model=? AND chunk_id IN ({marks})',
                                          (user_id, self.embedder.model_id, *chunk_ids)):
                        rows.append(row)
                        allowed += row['memory_id'] in allowed_ids
                if allowed >= target:
                    self.last_search = {'algorithm': 'lsh', 'radius': batch[-1][1], 'probes': used,
                                        'candidates': allowed, 'target': target, 'fallback': False}
                    return self.exact._rank(vector, rows, allowed_ids=allowed_ids, limit=limit, minimum=minimum)
        results = self.exact.search_vector(vector, user_id=user_id, allowed_ids=allowed_ids, limit=limit, minimum=minimum)
        self.last_search = {'algorithm': 'exact_fallback', 'radius': 2, 'probes': used, 'candidates': allowed,
                            'target': target, 'fallback': True}
        return results
