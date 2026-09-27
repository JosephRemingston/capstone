"""Rebuildable SQLite indexes. JSONL remains the authoritative observation log."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import struct

from .embeddings import Embedder, normalized
from .models import MemoryRecord

SCHEMA = '''
CREATE TABLE IF NOT EXISTS index_state (kind TEXT, user_id TEXT, fingerprint TEXT, PRIMARY KEY(kind,user_id));
CREATE TABLE IF NOT EXISTS vectors (
 user_id TEXT NOT NULL, model TEXT NOT NULL, chunk_id TEXT NOT NULL, memory_id TEXT NOT NULL,
 text TEXT NOT NULL, start_offset INTEGER, end_offset INTEGER, digest TEXT NOT NULL,
 dimension INTEGER NOT NULL, vector BLOB NOT NULL, PRIMARY KEY(user_id,model,chunk_id));
CREATE INDEX IF NOT EXISTS vector_owner ON vectors(user_id,model,memory_id);
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
