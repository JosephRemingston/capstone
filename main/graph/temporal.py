"""Provenance-preserving temporal property graph on SQLite.

Valid time describes accepted assertions; known time describes when the revision
was recorded. Both ranges are half-open. Traversals return evidence paths, never
invent transitive facts from mere connectivity.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
import json
import re

from ..domain.claims import extract_claims, normalized_value
from ..storage.indexes import IndexDatabase, digest
from ..domain.models import MemoryRecord, utc_now

NODE_TYPES = {'person', 'organization', 'project', 'location', 'task', 'concept', 'literal'}
PREDICATES = {
    'residence': ('lives_in', 'location'), 'employer': ('works_at', 'organization'),
    'manager': ('reports_to', 'person'), 'headquarters': ('based_in', 'location'),
    'works_on': ('works_on', 'project'), 'member_of': ('member_of', 'organization'),
    'owned_by': ('owned_by', 'organization'), 'depends_on': ('depends_on', 'project'),
    'knows': ('knows', 'person'), 'taste': ('likes', 'concept'), 'allergy': ('allergic_to', 'concept'),
    'name': ('named', 'literal'), 'occupation': ('occupation', 'concept'),
}


def iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError('Graph timestamps must include a timezone')
    return value.astimezone(timezone.utc).isoformat()


def parse_time(value, default=None):
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError('Graph timestamps must be ISO strings')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('Invalid graph timestamp') from exc
    iso(parsed)
    return parsed


def validate_graph_metadata(metadata: dict, created_at=None) -> None:
    entities, relations = metadata.get('entities', []), metadata.get('relations', [])
    if not isinstance(entities, list) or not isinstance(relations, list) or len(entities) > 100 or len(relations) > 200:
        raise ValueError('Graph entities/relations must be bounded lists')
    identifiers = {'self'}
    for entity in entities:
        if (not isinstance(entity, dict) or not isinstance(entity.get('id'), str) or not entity['id'].strip()
                or not isinstance(entity.get('type'), str) or entity.get('type') not in NODE_TYPES or not isinstance(entity.get('label'), str)
                or not entity['label'].strip() or entity['id'] in identifiers):
            raise ValueError('Entities require unique IDs, a supported type, and a label')
        identifiers.add(entity['id'])
        aliases = entity.get('aliases', [])
        if not isinstance(aliases, list) or any(not isinstance(alias, str) or not alias.strip() for alias in aliases):
            raise ValueError('Entity aliases must be nonempty strings')
    for relation in relations:
        if (not isinstance(relation, dict) or not isinstance(relation.get('subject'), str)
                or not isinstance(relation.get('object'), str) or relation.get('subject') not in identifiers
                or relation.get('object') not in identifiers or not isinstance(relation.get('predicate'), str)
                or not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', relation['predicate'])):
            raise ValueError('Relations require defined entity IDs and a snake_case predicate')
        if not isinstance(relation.get('positive', True), bool):
            raise ValueError('Relation polarity must be boolean')
        start, end = parse_time(relation.get('valid_from'), created_at), parse_time(relation.get('valid_to'))
        if start and end and end <= start:
            raise ValueError('Relationship valid_to must be after valid_from')


class TemporalGraph:
    def __init__(self, database: IndexDatabase):
        self.database = database

    def sync(self, revisions: list[MemoryRecord], *, user_id: str) -> dict:
        if not user_id:
            raise ValueError('A user_id is required')
        records = [record for record in revisions if record.user_id == user_id]
        fingerprint = digest(['graph-v2', [record.to_dict() for record in records]])
        with self.database.connect() as db:
            state = db.execute('SELECT fingerprint FROM index_state WHERE kind=? AND user_id=?', ('graph', user_id)).fetchone()
            if state and state[0] == fingerprint:
                return {'changed': False}
        nodes, aliases, edges = {}, {}, []

        def node(kind, key, label, known, names=(), resolve_alias=True):
            normalized = normalized_value(key)
            matches = {nid for (alias, nid), _ in aliases.items() if alias == normalized and nodes[nid]['kind'] == kind}
            if resolve_alias and len(matches) == 1:
                identifier = next(iter(matches))
            else:
                identifier = digest([user_id, kind, normalized])[:32]
            nodes.setdefault(identifier, {'id': identifier, 'kind': kind, 'label': label})
            for alias in (label, *names):
                pair = (normalized_value(alias), identifier)
                aliases[pair] = min(aliases.get(pair, known), known)
            return identifier

        def relationships(record, known):
            if record.role != 'user':
                return []
            validate_graph_metadata(record.source_metadata, record.created_at)
            speaker = record.source_metadata.get('speaker')
            self_key = normalized_value(speaker) if isinstance(speaker, str) and speaker else 'self'
            self_label = speaker if isinstance(speaker, str) and speaker else 'User'
            subject_self = node('person', self_key, self_label, known, resolve_alias=False)
            mapping = {'self': subject_self}
            for entity in record.source_metadata.get('entities', []):
                mapping[entity['id']] = node(entity['type'], entity['id'], entity['label'], known,
                                             entity.get('aliases', []), resolve_alias=False)
            found = []
            for relation in record.source_metadata.get('relations', []):
                found.append((mapping[relation['subject']], relation['predicate'], mapping[relation['object']],
                              relation.get('positive', True), parse_time(relation.get('valid_from'), record.created_at),
                              parse_time(relation.get('valid_to'))))
            for claim in extract_claims(record.content):
                kind = claim.get('subject_type', 'person')
                subject = subject_self if claim['subject'] == 'self' else node(
                    kind, claim['subject'].split(':', 1)[-1], claim['display_subject'], known)
                predicate, object_kind = PREDICATES.get(claim['predicate'], (re.sub(r'[^a-z0-9_]', '_', claim['predicate']), 'concept'))
                target = node(object_kind, claim['value'], claim['display_value'], known)
                if predicate == 'named' and claim['positive']:
                    pair = (normalized_value(claim['display_value']), subject)
                    aliases[pair] = min(aliases.get(pair, known), known)
                found.append((subject, predicate, target, claim['positive'], record.created_at, None))
            if record.category.value == 'task':
                task = node('task', record.id, record.content, known, resolve_alias=False)
                status = node('literal', record.task_status or 'active', record.task_status or 'active', known)
                found.extend([(subject_self, 'has_task', task, True, record.created_at, None),
                              (task, 'task_status', status, True, record.updated_at, None)])
                if record.due_at or record.next_due_at:
                    due = record.next_due_at or record.due_at
                    target = node('literal', iso(due), iso(due), known)
                    found.append((task, 'due_at', target, True, record.updated_at, None))
            return found

        # Process in recording order, so future aliases cannot rewrite old edges.
        grouped = defaultdict(list)
        for position, record in enumerate(records):
            grouped[record.id].append((position, record))
        material = {}
        for position, record in enumerate(records):
            known = iso(record.recorded_at or record.updated_at)
            material[position] = relationships(record, known)
        latest = {record.id: record for record in records}
        for source_id, versions in grouped.items():
            episodes = []
            previous_status = None
            previous_material = None
            for index, (position, record) in enumerate(versions):
                known_from = iso(record.recorded_at or record.updated_at)
                known_to = iso(versions[index + 1][1].recorded_at or versions[index + 1][1].updated_at) if index + 1 < len(versions) else None
                state = record.memory_status
                changed = previous_material is not None and material[position] != previous_material
                if state == 'active' and previous_status == 'active' and changed:
                    for episode in episodes:
                        if episode['open']:
                            episode['valid_to'] = min(episode['valid_to'] or iso(record.updated_at), iso(record.updated_at))
                            episode['open'] = False
                if state == 'active' and (previous_status != 'active' or changed):
                    for subject, predicate, target, positive, start, end in material[position]:
                        start = max(start, record.last_confirmed_at or start, record.updated_at if changed else start)
                        episodes.append({'subject': subject, 'predicate': predicate, 'object': target,
                                         'positive': positive, 'valid_from': iso(start), 'valid_to': iso(end) if end else None,
                                         'source_revision': digest(record.to_dict()), 'open': True})
                elif state == 'superseded' and previous_status == 'active':
                    resolution = record.conflict_resolution or {}
                    end = parse_time(resolution.get('recorded_at'), record.updated_at)
                    for episode in episodes:
                        if episode['open']:
                            episode['valid_to'] = min(episode['valid_to'] or iso(end), iso(end))
                            episode['open'] = False
                if state != 'consolidated':
                    for episode_number, episode in enumerate(episodes):
                        if episode['valid_to'] and episode['valid_to'] <= episode['valid_from']:
                            continue
                        evidence = record.evidence_ids or [source_id]
                        if any(key not in latest or latest[key].user_id != user_id for key in evidence):
                            raise ValueError('Graph evidence must belong to the indexed user')
                        edges.append((user_id, digest([source_id, position, episode_number]), episode['subject'],
                                      episode['predicate'], episode['object'], int(episode['positive']), source_id,
                                      episode['valid_from'], episode['valid_to'], known_from, known_to,
                                      record.confidence, json.dumps(evidence), episode['source_revision']))
                previous_status = state
                previous_material = material[position]
        with self.database.connect() as db:
            for table in ('edges', 'aliases', 'nodes'):
                db.execute(f'DELETE FROM {table} WHERE user_id=?', (user_id,))
            db.executemany('INSERT INTO nodes VALUES (?,?,?,?)', [(user_id, item['id'], item['kind'], item['label']) for item in nodes.values()])
            db.executemany('INSERT INTO aliases VALUES (?,?,?,?)', [(user_id, alias, nid, known) for (alias, nid), known in aliases.items()])
            db.executemany('INSERT INTO edges VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', edges)
            db.execute('INSERT OR REPLACE INTO index_state VALUES (?,?,?)', ('graph', user_id, fingerprint))
        return {'changed': True, 'nodes': len(nodes), 'edge_versions': len(edges),
                'legacy_time_fallbacks': sum(record.recorded_at is None for record in records)}

    def entities(self, query: str, *, user_id: str, known_at: datetime | None = None, exact=False) -> list[dict]:
        if not user_id:
            raise ValueError('A user_id is required')
        when = iso(known_at or utc_now())
        lowered = normalized_value(query)
        result = {}
        with self.database.connect() as db:
            for row in db.execute('SELECT a.alias,n.* FROM aliases a JOIN nodes n ON a.user_id=n.user_id AND a.node_id=n.id WHERE a.user_id=? AND a.known_from<=?', (user_id, when)):
                match = row['alias'] == lowered if exact else re.search(r'(?<!\w)' + re.escape(row['alias']) + r'(?!\w)', lowered)
                if match:
                    result[row['id']] = {'id': row['id'], 'type': row['kind'], 'label': row['label']}
        return sorted(result.values(), key=lambda row: row['id'])

    def relations(self, *, user_id: str, as_of: datetime | None = None, known_at: datetime | None = None,
                  entity_id: str | None = None, predicate: str | None = None, include_negative=True) -> list[dict]:
        if not user_id:
            raise ValueError('A user_id is required')
        valid, known = iso(as_of or utc_now()), iso(known_at or utc_now())
        sql = '''SELECT e.*,s.label subject_label,s.kind subject_type,o.label object_label,o.kind object_type
                 FROM edges e JOIN nodes s ON s.user_id=e.user_id AND s.id=e.subject
                 JOIN nodes o ON o.user_id=e.user_id AND o.id=e.object
                 WHERE e.user_id=? AND e.valid_from<=? AND (e.valid_to IS NULL OR e.valid_to>?)
                 AND e.known_from<=? AND (e.known_to IS NULL OR e.known_to>?)'''
        params = [user_id, valid, valid, known, known]
        if entity_id is not None:
            sql += ' AND (e.subject=? OR e.object=?)'
            params.extend([entity_id, entity_id])
        if predicate is not None:
            sql += ' AND e.predicate=?'
            params.append(predicate)
        if not include_negative:
            sql += ' AND e.positive=1'
        with self.database.connect() as db:
            rows = [dict(row) for row in db.execute(sql + ' ORDER BY e.subject,e.predicate,e.object,e.source_id', params)]
        for row in rows:
            row['positive'] = bool(row['positive'])
            row['evidence_ids'] = json.loads(row.pop('evidence'))
        return rows

    def traverse(self, seeds: list[str], *, user_id: str, as_of=None, known_at=None, max_hops=2,
                 max_paths=100, allowed_ids: set[str] | None = None, direction='both') -> list[dict]:
        if not 1 <= max_hops <= 4 or not 1 <= max_paths <= 1000 or direction not in {'both', 'outgoing', 'incoming'}:
            raise ValueError('Invalid traversal bounds or direction')
        relations = self.relations(user_id=user_id, as_of=as_of, known_at=known_at, include_negative=False)
        adjacency = defaultdict(list)
        for edge in relations:
            if allowed_ids is not None and edge['source_id'] not in allowed_ids:
                continue
            if direction in {'both', 'outgoing'}:
                adjacency[edge['subject']].append((edge['object'], edge))
            if direction in {'both', 'incoming'}:
                adjacency[edge['object']].append((edge['subject'], edge))
        queue = deque((seed, [seed], []) for seed in sorted(set(seeds)))
        result = []
        while queue and len(result) < max_paths:
            current, visited, path = queue.popleft()
            if len(path) >= max_hops:
                continue
            for target, edge in adjacency[current]:
                if target in visited:
                    continue
                steps = [*path, edge]
                result.append({'nodes': [*visited, target], 'edges': steps,
                               'source_ids': list(dict.fromkeys(step['source_id'] for step in steps))})
                queue.append((target, [*visited, target], steps))
                if len(result) == max_paths:
                    break
        return result
