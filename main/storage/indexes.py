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

    def search(self, query: str, *, user_id: str, allowed_ids: set[str], limit=20, minimum=0.35) -> list[dict]:
        if limit < 0 or not -1 <= minimum <= 1 or not user_id:
            raise ValueError('Invalid vector search parameters')
        if not query.strip() or not allowed_ids or limit == 0:
            return []
        vector = normalized(self.embedder.query(query), self.embedder.dimension)
        results = []
        with self.database.connect() as db:
            for row in db.execute('SELECT * FROM vectors WHERE user_id=? AND model=?', (user_id, self.embedder.model_id)):
                if row['memory_id'] not in allowed_ids:
                    continue
                if row['dimension'] != self.embedder.dimension:
                    raise ValueError('Vector index dimension mismatch; rebuild the index')
                stored = struct.unpack('<' + 'f' * row['dimension'], row['vector'])
                similarity = sum(a * b for a, b in zip(vector, stored))
                if similarity >= minimum:
                    results.append({'memory_id': row['memory_id'], 'chunk_id': row['chunk_id'],
                                    'text': row['text'], 'score': similarity,
                                    'start': row['start_offset'], 'end': row['end_offset']})
        return sorted(results, key=lambda item: (-item['score'], item['chunk_id']))[:limit]


class ApproximateVectorIndex:
    """Random-hyperplane LSH index for approximate cosine search.

    The existing SQLite vector table remains the source of vector data. This index
    adds multiple locality-sensitive hash tables, so large searches inspect only
    vectors that share one or more signatures with the query instead of scanning
    the complete corpus. It is deterministic for a given embedding model.
    """

    def __init__(self, database: IndexDatabase, embedder: Embedder, *, tables: int = 8, bits: int = 10,
                 seed: int = 1729):
        if not 1 <= tables <= 32 or not 1 <= bits <= 20:
            raise ValueError('LSH tables must be 1–32 and bits must be 1–20')
        self.database, self.embedder = database, embedder
        self.tables, self.bits, self.seed = tables, bits, seed
        self._planes = self._build_planes()
        self.exact = VectorIndex(database, embedder)

    def _build_planes(self):
        import random
        planes = []
        for table_id in range(self.tables):
            table = []
            for bit in range(self.bits):
                rng = random.Random(f'{self.seed}:{self.embedder.model_id}:{table_id}:{bit}')
                vector = [rng.gauss(0.0, 1.0) for _ in range(self.embedder.dimension)]
                norm = sum(value * value for value in vector) ** 0.5
                table.append([value / norm for value in vector])
            planes.append(table)
        return planes

    @staticmethod
    def _fingerprint(rows) -> str:
        return sha256(''.join(f"{row['chunk_id']}:{row['digest']}\n" for row in sorted(rows, key=lambda r: r['chunk_id'])).encode()).hexdigest()

    def _signature(self, vector, table_id: int) -> int:
        signature = 0
        for bit, plane in enumerate(self._planes[table_id]):
            dot = sum(a * b for a, b in zip(vector, plane))
            if dot >= 0:
                signature |= 1 << bit
        return signature

    def _rebuild_buckets(self, user_id: str, rows) -> None:
        with self.database.connect() as db:
            db.execute('DELETE FROM lsh_buckets WHERE user_id=? AND model=?', (user_id, self.embedder.model_id))
            inserts = []
            for row in rows:
                if row['dimension'] != self.embedder.dimension:
                    raise ValueError('Vector index dimension mismatch; rebuild the index')
                vector = struct.unpack('<' + 'f' * row['dimension'], row['vector'])
                for table_id in range(self.tables):
                    inserts.append((user_id, self.embedder.model_id, table_id, self._signature(vector, table_id), row['chunk_id']))
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
                         'indexed_chunks': len(rows)}
        return result

    def search(self, query: str, *, user_id: str, allowed_ids: set[str], limit=20, minimum=0.35) -> list[dict]:
        if limit < 0 or not -1 <= minimum <= 1 or not user_id:
            raise ValueError('Invalid vector search parameters')
        if not query.strip() or not allowed_ids or limit == 0:
            return []
        vector = normalized(self.embedder.query(query), self.embedder.dimension)
        candidate_chunks: set[str] = set()
        with self.database.connect() as db:
            # Multi-probe LSH: inspect the exact bucket first, then Hamming-distance
            # one and two buckets when necessary. This keeps the search approximate
            # while reducing false negatives caused by a single random projection.
            for radius in (0, 1, 2):
                for table_id in range(self.tables):
                    bucket = self._signature(vector, table_id)
                    masks = [0] if radius == 0 else []
                    if radius >= 1:
                        masks.extend(1 << bit for bit in range(self.bits))
                    if radius >= 2:
                        for first in range(self.bits):
                            for second in range(first + 1, self.bits):
                                masks.append((1 << first) | (1 << second))
                    for mask in masks:
                        candidate = bucket ^ mask
                        for row in db.execute(
                            'SELECT chunk_id FROM lsh_buckets WHERE user_id=? AND model=? AND table_id=? AND bucket=?',
                            (user_id, self.embedder.model_id, table_id, candidate),
                        ):
                            candidate_chunks.add(row['chunk_id'])
                if candidate_chunks:
                    break
            if not candidate_chunks:
                return []
            placeholders = ','.join('?' for _ in candidate_chunks)
            params = [user_id, self.embedder.model_id, *candidate_chunks]
            rows = db.execute(
                f'SELECT * FROM vectors WHERE user_id=? AND model=? AND chunk_id IN ({placeholders})', params
            )
            results = []
            for row in rows:
                if row['memory_id'] not in allowed_ids:
                    continue
                if row['dimension'] != self.embedder.dimension:
                    raise ValueError('Vector index dimension mismatch; rebuild the index')
                stored = struct.unpack('<' + 'f' * row['dimension'], row['vector'])
                similarity = sum(a * b for a, b in zip(vector, stored))
                if similarity >= minimum:
                    results.append({'memory_id': row['memory_id'], 'chunk_id': row['chunk_id'],
                                    'text': row['text'], 'score': similarity,
                                    'start': row['start_offset'], 'end': row['end_offset']})
        return sorted(results, key=lambda item: (-item['score'], item['chunk_id']))[:limit]
