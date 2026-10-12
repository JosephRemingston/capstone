"""Benchmark schema, loading, and structural validation shared by the builder and runners."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

QUERY_TYPES = (
    'semantic', 'episodic', 'temporal', 'historical', 'multi_hop', 'entity_relationship',
    'conflict', 'preference', 'procedural', 'recency', 'negative_absence', 'unanswerable',
)
SPLITS = ('dev', 'test')
GOLD_USER = 'gold-user'
DATASET_DIR = Path(__file__).resolve().parent / 'datasets'
DATASET_PATH = DATASET_DIR / 'gold_queries.json'
QUERY_KEYS = ('query_id', 'query', 'query_type', 'persona_id', 'split', 'relevant_memory_ids',
              'graded_relevance', 'supporting_memory_ids', 'expected_answer', 'answer_aliases', 'unanswerable')
MEMORY_KEYS = ('id', 'content', 'user_id', 'session_id', 'category', 'tier', 'created_at', 'recorded_at',
               'source_metadata')


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError(f'Timestamp must include a timezone: {value}')
    return parsed.astimezone(timezone.utc)


def validate_payload(payload: dict) -> None:
    """Structural checks every runner relies on. Content checks live in the builder."""
    for key in ('schema_version', 'label_provenance', 'splits', 'query_types', 'memories', 'queries'):
        if key not in payload:
            raise ValueError(f'Dataset is missing top-level key {key!r}')
    memories = {memory['id']: memory for memory in payload['memories']}
    if len(memories) != len(payload['memories']):
        raise ValueError('Memory IDs must be unique')
    for memory in payload['memories']:
        missing = [key for key in MEMORY_KEYS if key not in memory]
        if missing:
            raise ValueError(f'Memory {memory.get("id")} is missing {missing}')
        parse_time(memory['created_at'])
        parse_time(memory['recorded_at'])
    queries = payload['queries']
    if not 200 <= len(queries) <= 500:
        raise ValueError('The benchmark must contain 200-500 queries')
    ids = [query['query_id'] for query in queries]
    if len(set(ids)) != len(ids):
        raise ValueError('Query IDs must be unique')
    counts = Counter(query['query_type'] for query in queries)
    if set(counts) != set(QUERY_TYPES):
        raise ValueError(f'Every query type must be present: {sorted(set(QUERY_TYPES) - set(counts))}')
    for query in queries:
        missing = [key for key in QUERY_KEYS if key not in query]
        if missing:
            raise ValueError(f'Query {query.get("query_id")} is missing {missing}')
        grades = query['graded_relevance']
        if set(grades) != set(query['relevant_memory_ids']):
            raise ValueError(f'{query["query_id"]}: graded_relevance keys must equal relevant_memory_ids')
        if any(isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= 3 for grade in grades.values()):
            raise ValueError(f'{query["query_id"]}: grades must be integers 0-3')
        if any(key not in memories for key in grades) or any(key not in memories for key in query['supporting_memory_ids']):
            raise ValueError(f'{query["query_id"]}: references an unknown memory')
        if not set(query['supporting_memory_ids']) <= set(grades):
            raise ValueError(f'{query["query_id"]}: supporting memories must be relevant')
        if query['unanswerable'] != (not grades):
            raise ValueError(f'{query["query_id"]}: unanswerable must mean no gold evidence')
        if query['split'] not in SPLITS:
            raise ValueError(f'{query["query_id"]}: unknown split {query["split"]!r}')
        if ('as_of' in query) != (query['query_type'] == 'historical'):
            raise ValueError(f'{query["query_id"]}: as_of is required for, and only for, historical queries')
        if 'as_of' in query:
            parse_time(query['as_of'])
    dev = {query['persona_id'] for query in queries if query['split'] == 'dev'}
    test = {query['persona_id'] for query in queries if query['split'] == 'test'}
    if dev & test:
        raise ValueError(f'Personas appear in both splits: {sorted(dev & test)}')


def load_dataset(path: Path | str = DATASET_PATH, split: str = 'test') -> tuple[dict, list[dict]]:
    """Return the full payload and the queries of one split ('dev', 'test', or 'all')."""
    payload = json.loads(Path(path).read_text())
    validate_payload(payload)
    if split == 'all':
        selected = list(payload['queries'])
    elif split in SPLITS:
        selected = [query for query in payload['queries'] if query['split'] == split]
    else:
        raise ValueError(f'Unknown split {split!r}')
    if len(selected) < 50 or {query['query_type'] for query in selected} != set(QUERY_TYPES):
        raise ValueError(f'Split {split!r} needs at least 50 queries covering every query type')
    return payload, selected
