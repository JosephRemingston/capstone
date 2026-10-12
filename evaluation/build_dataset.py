"""Compile Claude-authored persona seeds into the frozen benchmark file.

    python -m evaluation.build_dataset            # write evaluation/datasets/gold_queries.json
    python -m evaluation.build_dataset --check    # verify the file equals a fresh build
    python -m evaluation.build_dataset --lint     # list queries that copy their evidence wording

Seed format (evaluation/datasets/seeds/pNN_name.json): a persona header, namespaced-later
entity definitions, dated sessions, memories (key, session, category, text, optional
relations/due), and queries (type, query, gold {key: grade}, supporting, answer, aliases,
as_of for historical). Content validators enforce the properties the runners rely on.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import re

from main.domain.claims import extract_claims
from main.domain.models import MemoryRecord
from main.graph.temporal import validate_graph_metadata
from main.retrieval.rag import graph_predicates
from main.storage.indexes import document_text

from .metrics import contains_phrase, normalize
from .schema import DATASET_DIR, DATASET_PATH, GOLD_USER, QUERY_TYPES, parse_time, validate_payload

SEED_DIR = DATASET_DIR / 'seeds'
AUTHORED_ON = '2026-10-12'
PROVENANCE = f'authored by Claude (AI) on {AUTHORED_ON}; not human-validated'
RESEARCH_QUESTION = ('How much does adding temporal and relational graph retrieval improve '
                     'vector-based conversational memory retrieval and RAG performance?')
PERSONAS, DEV_PERSONAS, MIN_MEMORIES, QUERIES_PER_TYPE = 12, 3, 440, 30
CATEGORIES = {'semantic', 'episodic', 'procedural', 'preference', 'task'}
CLOSED_CLAIMS = {'residence', 'employer', 'occupation'}


def iso_day(value: str) -> str:
    """Accept YYYY-MM-DD (midnight UTC) or a full timezone-aware ISO timestamp."""
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        value += 'T00:00:00+00:00'
    return parse_time(value).isoformat()


def load_seeds(seed_dir: Path = SEED_DIR) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(seed_dir.glob('p*.json'))]


def compile_persona(seed: dict) -> tuple[list[dict], list[dict]]:
    persona = seed['persona']
    pid, name = persona['id'], persona['name']
    entities = seed.get('entities', {})

    def ref(identifier: str) -> str:
        return identifier if identifier == 'self' else f'{pid}:{identifier}'

    memories, order = [], defaultdict(int)
    for item in seed['memories']:
        session_start = datetime.fromisoformat(iso_day(seed['sessions'][item['session']]))
        created = (session_start + timedelta(minutes=4 * order[item['session']])).isoformat()
        order[item['session']] += 1
        metadata = {'speaker': name, 'session_date': session_start.date().isoformat()}
        relations = []
        for relation in item.get('relations', []):
            compiled = {'subject': ref(relation['subject']), 'predicate': relation['predicate'],
                        'object': ref(relation['object']), 'positive': relation.get('positive', True)}
            for key in ('valid_from', 'valid_to'):
                if relation.get(key):
                    compiled[key] = iso_day(relation[key])
            relations.append(compiled)
        if relations:
            used = sorted({r[k] for r in item['relations'] for k in ('subject', 'object')} - {'self'})
            metadata['entities'] = [{'id': ref(key), 'type': entities[key]['type'], 'label': entities[key]['label'],
                                     **({'aliases': entities[key]['aliases']} if entities[key].get('aliases') else {})}
                                    for key in used]
            metadata['relations'] = relations
        memory = {'id': f'{pid}-{item["key"]}', 'content': item['text'], 'user_id': GOLD_USER,
                  'session_id': f'{pid}-{item["session"]}', 'role': 'user', 'category': item['category'],
                  'tier': 'short_term' if item['category'] == 'task' else 'long_term',
                  'created_at': created, 'updated_at': created, 'recorded_at': created,
                  'confidence': 1.0, 'importance_score': 0.8, 'source_metadata': metadata}
        if item['category'] == 'task':
            memory['task_status'] = 'active'
            if item.get('due'):
                memory['due_at'] = iso_day(item['due'])
        memories.append(memory)

    queries, numbers = [], Counter()
    for item in seed['queries']:
        numbers[item['type']] += 1
        grades = {f'{pid}-{key}': grade for key, grade in item.get('gold', {}).items()}
        query = {'query_id': f'q-{pid}-{item["type"]}-{numbers[item["type"]]}', 'query': item['query'],
                 'query_type': item['type'], 'persona_id': pid, 'split': persona['split'],
                 'relevant_memory_ids': sorted(grades, key=lambda key: (-grades[key], key)),
                 'graded_relevance': grades,
                 'supporting_memory_ids': [f'{pid}-{key}' for key in item.get('supporting', [])],
                 'expected_answer': item.get('answer'), 'answer_aliases': item.get('aliases', []),
                 'unanswerable': not grades}
        if item.get('as_of'):
            query['as_of'] = iso_day(item['as_of'])
        queries.append(query)
    return memories, queries


def build(seed_dir: Path = SEED_DIR) -> dict:
    seeds = load_seeds(seed_dir)
    memories, queries = [], []
    for seed in seeds:
        persona_memories, persona_queries = compile_persona(seed)
        memories += persona_memories
        queries += persona_queries
    payload = {
        'schema_version': 2, 'research_question': RESEARCH_QUESTION, 'label_provenance': PROVENANCE,
        'authored_on': AUTHORED_ON, 'query_types': list(QUERY_TYPES),
        'splits': {split: sorted(s['persona']['id'] for s in seeds if s['persona']['split'] == split)
                   for split in ('dev', 'test')},
        'personas': {s['persona']['id']: s['persona']['name'] for s in seeds},
        'memories': sorted(memories, key=lambda m: m['id']),
        'queries': sorted(queries, key=lambda q: q['query_id']),
    }
    validate_content(payload, seeds)
    return payload


def validate_content(payload: dict, seeds: list[dict]) -> None:
    """Raise ValueError listing every content problem found."""
    errors = []
    try:
        validate_payload(payload)
    except ValueError as exc:
        errors.append(str(exc))
    memories = {m['id']: m for m in payload['memories']}
    records = {}
    for memory in payload['memories']:
        if memory['category'] not in CATEGORIES:
            errors.append(f'{memory["id"]}: unsupported category {memory["category"]}')
        try:
            validate_graph_metadata(memory['source_metadata'], parse_time(memory['created_at']))
            records[memory['id']] = MemoryRecord.from_dict(memory)
        except (ValueError, KeyError) as exc:
            errors.append(f'{memory["id"]}: invalid memory ({exc})')
        relations = memory['source_metadata'].get('relations', [])
        if any(r.get('valid_to') for r in relations):
            leaked = {c['predicate'] for c in extract_claims(memory['content'])} & CLOSED_CLAIMS
            if leaked:
                errors.append(f'{memory["id"]}: text yields an open {sorted(leaked)} claim that contradicts its closed relation')
    if len(seeds) != PERSONAS:
        errors.append(f'Expected {PERSONAS} personas, found {len(seeds)}')
    if sum(s['persona']['split'] == 'dev' for s in seeds) != DEV_PERSONAS:
        errors.append(f'Expected {DEV_PERSONAS} dev personas')
    if len(memories) < MIN_MEMORIES:
        errors.append(f'Expected at least {MIN_MEMORIES} memories, found {len(memories)}')
    totals = Counter(q['query_type'] for q in payload['queries'])
    for kind in QUERY_TYPES:
        if totals[kind] != QUERIES_PER_TYPE:
            errors.append(f'{kind}: {totals[kind]} queries, expected {QUERIES_PER_TYPE}')
    names = payload['personas']
    for pid in names:
        persona_queries = [q for q in payload['queries'] if q['persona_id'] == pid]
        counts = Counter(q['query_type'] for q in persona_queries)
        if len(persona_queries) != 30 or any(counts[kind] not in (2, 3) for kind in QUERY_TYPES):
            errors.append(f'{pid}: needs 30 queries with 2-3 per type, has {dict(counts)}')
    for query in payload['queries']:
        qid, text = query['query_id'], query['query']
        name = names[query['persona_id']]
        if not re.search(rf'\b{re.escape(name)}\b', text):
            errors.append(f'{qid}: query must name the persona {name!r}')
        if len(text.split()) < 6:
            errors.append(f'{qid}: query needs at least 6 words')
        if any(not key.startswith(query['persona_id'] + '-') for key in query['graded_relevance']):
            errors.append(f'{qid}: gold evidence must belong to its persona')
        if query['unanswerable']:
            if query['expected_answer'] is not None or query['answer_aliases']:
                errors.append(f'{qid}: unanswerable queries carry no answer')
            continue
        if not query['supporting_memory_ids']:
            errors.append(f'{qid}: answerable queries need supporting memories')
            continue
        forms = [query['expected_answer'], *query['answer_aliases']]
        if not query['expected_answer'] or not any(
                contains_phrase(document_text(records[key]), form)
                for key in query['supporting_memory_ids'] if key in records for form in forms):
            errors.append(f'{qid}: no answer form appears in a supporting memory')
        support = [memories[key] for key in query['supporting_memory_ids'] if key in memories]
        if query['query_type'] == 'historical':
            when = parse_time(query['as_of'])
            valid = any(parse_time(r.get('valid_from', m['created_at'])) <= when
                        and (not r.get('valid_to') or when < parse_time(r['valid_to']))
                        for m in support for r in m['source_metadata'].get('relations', []))
            if not valid:
                errors.append(f'{qid}: no supporting relation is valid at as_of')
        if query['query_type'] == 'multi_hop':
            if len(support) < 2:
                errors.append(f'{qid}: multi-hop needs two supporting memories')
                continue
            entity_sets = [{r[k] for r in m['source_metadata'].get('relations', []) for k in ('subject', 'object')} - {'self'}
                           for m in support]
            if not set.intersection(*entity_sets):
                errors.append(f'{qid}: supporting memories share no entity, so no graph path joins them')
            used = {r['predicate'] for m in support for r in m['source_metadata'].get('relations', [])}
            missing = used - graph_predicates(text)
            if missing:
                errors.append(f'{qid}: query does not cue graph predicates {sorted(missing)}')
    if errors:
        raise ValueError(f'{len(errors)} dataset problem(s):\n  ' + '\n  '.join(errors))


def lint(payload: dict, top: int = 15) -> list[tuple[float, str, str]]:
    """Queries whose wording overlaps most with their best gold memory (possible leakage)."""
    stop = set('the a an of to in on at for is was what which who where when does did do how has her his their '
               'priya s with and or by from that this it be'.split())
    memories = {m['id']: m for m in payload['memories']}
    scored = []
    for query in payload['queries']:
        terms = set(normalize(query['query']).split()) - stop - set(normalize(payload['personas'][query['persona_id']]).split())
        for key in query['relevant_memory_ids']:
            words = set(normalize(memories[key]['content']).split()) - stop
            if terms:
                scored.append((len(terms & words) / len(terms), query['query_id'], key))
    return sorted(scored, reverse=True)[:top]


def serialize(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true', help='fail if the saved file differs from a fresh build')
    parser.add_argument('--lint', action='store_true', help='print queries with the highest wording overlap')
    parser.add_argument('--output', type=Path, default=DATASET_PATH)
    args = parser.parse_args()
    payload = build()
    text = serialize(payload)
    if args.check:
        if not args.output.exists() or args.output.read_text() != text:
            raise SystemExit(f'{args.output} is stale; run python -m evaluation.build_dataset')
        print('Dataset file matches a fresh build.')
    else:
        args.output.write_text(text)
    if args.lint:
        for overlap, qid, key in lint(payload):
            print(f'{overlap:.2f}  {qid}  <- {key}')
    counts = Counter(q['split'] for q in payload['queries'])
    print(json.dumps({'memories': len(payload['memories']), 'queries': len(payload['queries']),
                      'split_queries': dict(counts), 'sha256': hashlib.sha256(text.encode()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
