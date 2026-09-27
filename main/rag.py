"""Hybrid memory retrieval and bounded, provenance-preserving context."""
from __future__ import annotations

from dataclasses import replace
import json
import re

from .embeddings import FastEmbedder
from .graph import TemporalGraph, iso
from .indexes import IndexDatabase, VectorIndex, document_text, chunks, digest
from .models import utc_now
from .retrieval import tokenize

MODES = ('keyword', 'semantic', 'graph', 'hybrid', 'recency')


class MemoryRAG:
    def __init__(self, store, *, embedder=None, index_path=None):
        self.store = store
        self.database = IndexDatabase(index_path or store.path.with_suffix('.index.sqlite3'))
        self.graph = TemporalGraph(self.database)
        self._embedder = embedder

    @property
    def vectors(self):
        if self._embedder is None:
            self._embedder = FastEmbedder()
        return VectorIndex(self.database, self._embedder)

    def sync(self, *, user_id, semantic=True):
        result = {'graph': self.graph.sync(self.store._read_log(), user_id=user_id)}
        if semantic:
            result['vectors'] = self.vectors.sync(self.store.list(user_id=user_id), user_id=user_id)
        return result

    def candidates(self, *, user_id, as_of=None, known_at=None, **filters):
        if not user_id:
            raise ValueError('A user_id is required for RAG')
        if as_of is None and known_at is None:
            return self.store.list(user_id=user_id, **filters)
        # Historical RAG uses graph assertions only: their validity is explicit.
        # Excluding undated free text avoids presenting present-day text as past truth.
        self.graph.sync(self.store._read_log(), user_id=user_id)
        valid, known = as_of or utc_now(), known_at or utc_now()
        edges = self.graph.relations(user_id=user_id, as_of=valid, known_at=known)
        sources = {edge['source_id'] for edge in edges}
        revisions = {edge['source_revision'] for edge in edges}
        snapshots = {}
        for record in self.store._read_log():
            if (record.user_id == user_id and (record.recorded_at or record.updated_at) <= known
                    and digest(record.to_dict()) in revisions):
                snapshots[record.id] = record
        records = [replace(record, summary=None) for identifier, record in snapshots.items()
                   if identifier in sources and (record.expires_at is None or record.expires_at > valid)
                   and record.task_status not in {'completed', 'cancelled'}]
        return self.store._filter(records, user_id=user_id, session_id=filters.get('session_id'),
                                  category=filters.get('category'), tier=filters.get('tier'))

    def search(self, query, *, user_id, mode='hybrid', limit=5, as_of=None, known_at=None,
               max_hops=2, **filters):
        for timestamp in (as_of, known_at):
            if timestamp is not None:
                iso(timestamp)
        if mode not in MODES or not isinstance(limit, int) or not 0 <= limit <= 100:
            raise ValueError('Invalid retrieval mode or limit (0–100)')
        if not user_id:
            raise ValueError('A user_id is required for RAG')
        if not query.strip() or not limit:
            return []
        records = self.candidates(user_id=user_id, as_of=as_of, known_at=known_at, **filters)
        by_id = {record.id: record for record in records}
        allowed = set(by_id)
        pool = max(20, limit * 4)
        channels, dense, paths = {}, {}, []
        if mode == 'recency':
            channels['recency'] = [r.id for r in sorted(records, key=lambda r: (r.created_at, r.id), reverse=True)[:pool]]
        if mode in {'keyword', 'hybrid'}:
            # Speaker/date metadata are part of the retrieval document, not labels.
            lexical = [replace(r, content=document_text(r), summary=None) for r in records]
            channels['keyword'] = [r.id for r in self.store.ranker.rank(query, lexical, now=as_of)[:pool]]
        if mode in {'semantic', 'hybrid'}:
            vectors = self.vectors
            # Sync the whole current user's corpus, then filter results. A restrictive
            # session query must not evict other sessions from the persistent index.
            corpus = records if as_of or known_at else list({r.id: r for r in [*self.store.list(user_id=user_id), *records]}.values())
            vectors.sync(corpus, user_id=user_id)
            for hit in vectors.search(query, user_id=user_id, allowed_ids=allowed, limit=pool * 3):
                dense.setdefault(hit['memory_id'], hit)
            channels['semantic'] = list(dense)[:pool]
        if mode in {'graph', 'hybrid'}:
            self.graph.sync(self.store._read_log(), user_id=user_id)
            seeds = self.graph.entities(query, user_id=user_id, known_at=known_at)
            if re.search(r'\b(my|me|i)\b', query, re.I):
                seeds += self.graph.entities('User', user_id=user_id, known_at=known_at, exact=True)
            paths = self.graph.traverse([s['id'] for s in seeds], user_id=user_id,
                                       as_of=as_of, known_at=known_at, max_hops=max_hops, allowed_ids=allowed)
            # Negative assertions are useful direct evidence, but are never traversed
            # as positive links when constructing multi-hop paths.
            seed_ids = {s['id'] for s in seeds}
            negatives = [e['source_id'] for e in self.graph.relations(user_id=user_id, as_of=as_of, known_at=known_at)
                         if not e['positive'] and e['source_id'] in allowed
                         and (e['subject'] in seed_ids or e['object'] in seed_ids)]
            channels['graph'] = list(dict.fromkeys([*negatives, *(identifier for path in paths for identifier in path['source_ids'])]))[:pool]
        weights = {'keyword': .3, 'semantic': .5, 'graph': .2, 'recency': 1.0}
        scores, details = {}, {}
        for channel, identifiers in channels.items():
            for rank, identifier in enumerate(identifiers, 1):
                contribution = (weights[channel] if mode == 'hybrid' else 1) / (60 + rank)
                scores[identifier] = scores.get(identifier, 0) + contribution
                details.setdefault(identifier, {})[channel] = {'rank': rank, 'contribution': contribution}
        result = []
        for identifier in sorted(scores, key=lambda key: (-scores[key], key))[:limit]:
            record = by_id[identifier]
            if identifier in dense:
                snippet = dense[identifier]['text']
            else:
                terms = tokenize(query)
                snippets = chunks(record)
                snippet = max(snippets, key=lambda c: len(terms & tokenize(c.text))).text if snippets else ''
            result.append({'memory_id': identifier, 'text': snippet, 'score': scores[identifier],
                           'signals': details[identifier], 'record': record.to_dict(),
                           'paths': [path for path in paths if identifier in path['source_ids']]})
        return result

    def context(self, query, *, user_id, budget=10000, **search_options):
        if not 256 <= budget <= 100000:
            raise ValueError('Context budget must be between 256 and 100000 characters')
        hits = self.search(query, user_id=user_id, **search_options)
        sources, blocks, used = {}, [], 0
        for hit in hits:
            record = hit['record']
            block = {'id': hit['memory_id'], 'observed_at': record['created_at'],
                     'text': hit['text']}
            encoded = json.dumps(block, ensure_ascii=False)
            # Skip whole chunks that do not fit; never cut a statement midway.
            if used + len(encoded) + 1 > budget:
                continue
            sources[hit['memory_id']] = block
            blocks.append(encoded)
            used += len(encoded) + 1
        # Include a path only when every supporting source fits in the context.
        paths, seen = [], set()
        for hit in hits:
            for path in hit['paths']:
                signature = tuple(edge['id'] for edge in path['edges'])
                if signature in seen or not all(key in sources for key in path['source_ids']):
                    continue
                compact = {'path': [{'subject': e['subject_label'], 'predicate': e['predicate'],
                                     'object': e['object_label'], 'source_id': e['source_id'],
                                     'valid_from': e['valid_from'], 'valid_to': e['valid_to']}
                                    for e in path['edges']]}
                encoded = json.dumps(compact, ensure_ascii=False)
                if used + len(encoded) + 1 <= budget:
                    blocks.append(encoded)
                    paths.append(compact)
                    seen.add(signature)
                    used += len(encoded) + 1
        return {'query': query, 'sources': sources, 'paths': paths,
                'text': '\n'.join(blocks), 'characters': len('\n'.join(blocks)),
                'as_of': iso(search_options.get('as_of') or utc_now()),
                'known_at': iso(search_options.get('known_at') or utc_now()),
                'retrieved': len(hits), 'historical': bool(search_options.get('as_of') or search_options.get('known_at'))}
