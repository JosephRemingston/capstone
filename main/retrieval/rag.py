"""Hybrid memory retrieval and bounded, provenance-preserving context."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
import math
import os
import re
import time

from .embeddings import FastEmbedder
from ..graph import TemporalGraph, iso
from ..storage.indexes import IndexDatabase, VectorIndex, ApproximateVectorIndex, document_text, chunks, digest
from ..domain.models import utc_now
from .ranker import tokenize
from .reranker import FastEmbedCrossEncoderReranker
from ..storage.locking import locked, store_lock
from ..storage.cleanup import register_index

MODES = ('keyword', 'semantic', 'graph', 'hybrid', 'recency')
MAX_POOL = 200
DEFAULT_POOL_FLOOR = 100


def default_pool(limit, override=None, *, rerank=True):
    """Fused candidates handed to the reranker when the caller gives no candidate_limit.

    LoCoMo audit: a pool of 20 holds gold evidence 69.6% of the time, 100 holds 91.9%.
    Without reranking the fused order is final, and longer channel lists only add deep
    RRF votes (benchmark dev top-1 fell from 63.9% to 51.8%), so it keeps max(20, 4*limit).
    Always within [limit, MAX_POOL], so every valid limit (0-100) has a valid pool.
    """
    if not rerank:
        return min(MAX_POOL, max(20, 4 * limit))
    pool = max(DEFAULT_POOL_FLOOR, 4 * limit) if override is None else override
    return min(MAX_POOL, max(limit, pool))


def options_from_env(environ=None) -> dict:
    """MemoryRAG keyword options from COGNIMEM_* variables (shared by the CLI and the API)."""
    environ = os.environ if environ is None else environ

    def flag(name, default):
        raw = environ.get(name, '').strip().lower()
        if not raw:
            return default
        if raw in {'1', 'true', 'yes', 'on'}:
            return True
        if raw in {'0', 'false', 'no', 'off'}:
            return False
        raise ValueError(f'{name} must be true or false')

    def integer(name, default, maximum=None):
        raw = environ.get(name, '').strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f'{name} must be an integer') from None
        return _positive_int(value, name, maximum=maximum)

    return {'ann_enabled': flag('COGNIMEM_ANN_ENABLED', False),
            'ann_exact_threshold': integer('COGNIMEM_ANN_EXACT_THRESHOLD', 5000),
            'candidate_pool': integer('COGNIMEM_CANDIDATE_POOL', None, MAX_POOL),
            'model_threads': integer('COGNIMEM_MODEL_THREADS', None)}


def _positive_int(value, name, *, maximum=None, optional=False):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or (maximum and value > maximum):
        bound = f' between 1 and {maximum}' if maximum else ''
        raise ValueError(f'{name} must be a positive integer{bound}')
    return value


def graph_predicates(query):
    """Return graph predicates explicitly suggested by a natural-language query."""
    lowered = query.casefold()
    predicates = set()
    cues = (
        (r'\b(live|lives|lived|home|residence|city)\b', {'lives_in'}),
        (r'\b(work|works|worked|employer|company|office|based|headquarter)\w*\b', {'works_at', 'based_in'}),
        (r'\b(report(?:s|ed)? to|manager|supervisor)\b', {'reports_to'}),
        (r'\b(prefer|prefers|preference|like|likes|favorite|favourite)\w*\b', {'likes'}),
        (r'\b(allerg|allergy)\w*\b', {'allergic_to'}),
        (r'\b(occupation|profession|job|role)\b', {'occupation'}),
        (r'\b(task|deadline|due)\b', {'has_task', 'task_status', 'due_at'}),
        (r'\b(project|depends|dependency)\b', {'works_on', 'depends_on'}),
        (r'\b(member|membership|belongs)\b', {'member_of'}),
        (r'\b(owner|owned)\b', {'owned_by'}),
    )
    for pattern, values in cues:
        if re.search(pattern, lowered):
            predicates.update(values)
    return predicates


_HISTORY_CUES = re.compile(
    r'\b(before|originally|original|initially|at first|first (?:said|say|says|told|mentioned|planned)|'
    r'previously|previous|prior|used to|earlier|formerly|former|back in|back then|ago|last year)\b')
_PAST_YEAR = re.compile(r'\b(?:in|during) ((?:19|20)\d\d)\b')
_CURRENT_CUES = re.compile(
    r'\b(now|currently|current|these days|nowadays|at the moment|right now|latest|most recent|most recently|'
    r'newest|recent|recently|lately|any ?more|still|today|in the end|end(?:ed)? up|turn(?:ed)? out|'
    r'after the (?:\w+ )?(?:change|update|correction|switch|move))\b')
# Words that never identify what two statements are about (function words, time cues).
_TOPIC_STOPWORDS = frozenset('''
a an the and or but if of to in on at by for with from as is are was were be been being am do does did done have has
had having i me my mine we our you your he him his she her they them their it its this that these those there here what
which who whom whose where when why how whats s t d ll m re ve not no yes so very just about into over after before up
down out off again further then once than too can could will would shall should may might must also only own same such
any all both each few more most other some nor now currently current days nowadays moment right latest recent recently
lately newest anymore still today end ended turn turned change update correction switch move originally original
initially first previously previous prior used earlier formerly former back ago last year during tell told say said
mentioned like get got go went'''.split())
# 'versions' (prefer the newer/older of two same-speaker statements on a cue word) is opt-in: its
# benchmark gains vanished under paraphrase and it changed 10 LoCoMo rankings with 0 wins, 2 losses.
ALL_STRUCTURE_STEPS = frozenset({'graph', 'validity', 'versions', 'chain'})
STRUCTURE_STEPS = frozenset({'graph', 'validity', 'chain'})
VERSION_MEMBERS = 10


def time_intent(query, now=None):
    """Classify the temporal reading a query asks for: 'history', 'current', or 'neutral'.

    History cues ask for an earlier state ("originally", "before", "in 2024"); current cues ask
    for the latest one ("now", "these days", "most recent"). Both or neither means neutral.
    """
    lowered = query.casefold()
    year = (now or utc_now()).year
    history = bool(_HISTORY_CUES.search(lowered)) or any(
        int(match) < year for match in _PAST_YEAR.findall(lowered))
    current = bool(_CURRENT_CUES.search(lowered))
    if history != current:
        return 'history' if history else 'current'
    return 'neutral'


def _topic_terms(text):
    return {term for term in tokenize(text)
            if len(term) > 1 and not term.isdigit() and term not in _TOPIC_STOPWORDS}


class MemoryRAG:
    # Post-rerank structural ordering (hybrid mode with the graph channel and an applied reranker).
    # A candidate is "relevant" when its cross-encoder logit is within structure_band of the best one;
    # only relevant candidates can be promoted by graph evidence or reordered as statement versions.
    structure_steps = STRUCTURE_STEPS
    structure_band = 3.0

    def __init__(
        self,
        store,
        *,
        embedder=None,
        index_path=None,
        reranker='auto',
        ann_enabled=False,
        ann_exact_threshold=5000,
        candidate_pool=None,
        model_threads=None,
    ):
        self.store = store
        with store_lock(store.path):
            self.database = IndexDatabase(index_path or store.path.with_suffix('.index.sqlite3'))
            register_index(store.path, self.database.path)
        self.graph = TemporalGraph(self.database)
        self._embedder = embedder
        self._reranker = reranker
        if not isinstance(ann_enabled, bool):
            raise ValueError('ann_enabled must be a boolean')
        self.ann_enabled = ann_enabled
        self.ann_exact_threshold = _positive_int(ann_exact_threshold, 'ann_exact_threshold')
        self.candidate_pool = _positive_int(candidate_pool, 'candidate_pool', maximum=MAX_POOL, optional=True)
        self.model_threads = _positive_int(model_threads, 'model_threads', optional=True)
        self._indexes = {}

    @property
    def reranker(self):
        if self._reranker == 'auto':
            self._reranker = FastEmbedCrossEncoderReranker(threads=self.model_threads)
        return self._reranker

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = FastEmbedder() if self.model_threads is None else FastEmbedder(threads=self.model_threads)
        return self._embedder

    def vector_index(self, corpus_size=None):
        """Exact scan unless ANN is enabled and this user's corpus reaches the threshold.

        Index objects are cached per (approximate, embedding model): LSH hyperplanes
        are built once per process instead of on every query.
        """
        embedder = self.embedder
        approximate = bool(self.ann_enabled and corpus_size is not None and corpus_size >= self.ann_exact_threshold)
        key = (approximate, embedder.model_id)
        index = self._indexes.get(key)
        if index is None or index.embedder is not embedder:
            index = (ApproximateVectorIndex if approximate else VectorIndex)(self.database, embedder)
            self._indexes[key] = index
        return index

    @property
    def vectors(self):
        """Backward-compatible accessor; the store is only counted when ANN is enabled."""
        return self.vector_index(len(self.store.list()) if self.ann_enabled else None)

    @locked
    def sync(self, *, user_id, semantic=True):
        result = {'graph': self.graph.sync(self.store._read_log(), user_id=user_id)}
        if semantic:
            records = self.store.list(user_id=user_id)
            result['vectors'] = self.vector_index(len(records)).sync(records, user_id=user_id)
        return result

    @locked
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

    @locked
    def search(self, query, *, user_id, mode='hybrid', limit=5, as_of=None, known_at=None,
               max_hops=2, rerank=None, candidate_limit=None, channels=None, fusion_limit=None, **filters):
        search_started = time.perf_counter()
        for timestamp in (as_of, known_at):
            if timestamp is not None:
                iso(timestamp)
        if mode not in MODES or isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 100:
            raise ValueError('Invalid retrieval mode or limit (0–100)')
        if fusion_limit is not None and (isinstance(fusion_limit, bool) or not isinstance(fusion_limit, int)
                                         or not 0 <= fusion_limit <= 100):
            raise ValueError('fusion_limit must be an integer between 0 and 100')
        if rerank not in (None, True, False):
            raise ValueError('rerank must be true, false, or null')
        should_rerank = (mode == 'hybrid') if rerank is None else rerank
        pool = (default_pool(limit, self.candidate_pool, rerank=should_rerank)
                if candidate_limit is None else candidate_limit)
        if (isinstance(pool, bool) or not isinstance(pool, int) or not limit <= pool <= MAX_POOL):
            raise ValueError(f'candidate_limit must be an integer between limit and {MAX_POOL}')
        defaults = {'keyword': ('keyword',), 'semantic': ('semantic',),
                    'graph': ('graph',), 'hybrid': ('keyword', 'semantic', 'graph'),
                    'recency': ('recency',)}
        active_channels = defaults[mode] if channels is None else tuple(channels)
        if (channels is not None and mode != 'hybrid') or not active_channels or len(set(active_channels)) != len(active_channels):
            raise ValueError('Custom channels require hybrid mode and must be unique')
        # Only caller-supplied channel lists are restricted; built-in modes (recency) keep their defaults.
        if channels is not None and any(channel not in {'keyword', 'semantic', 'graph'} for channel in active_channels):
            raise ValueError('Custom channels may contain keyword, semantic, and graph')
        if not user_id:
            raise ValueError('A user_id is required for RAG')
        if not query.strip() or not limit:
            return []
        records = self.candidates(user_id=user_id, as_of=as_of, known_at=known_at, **filters)
        timings = {'filter': time.perf_counter() - search_started, 'keyword': 0.0,
                   'vector': 0.0, 'graph': 0.0, 'fusion': 0.0, 'reranker': 0.0}
        by_id = {record.id: record for record in records}
        allowed = set(by_id)
        channels, dense, paths, vector_diagnostics = {}, {}, [], None
        if 'recency' in active_channels:
            channels['recency'] = [r.id for r in sorted(records, key=lambda r: (r.created_at, r.id), reverse=True)[:pool]]
        if 'keyword' in active_channels:
            stage_started = time.perf_counter()
            # Speaker/date metadata are part of the retrieval document, not labels.
            lexical = [replace(r, content=document_text(r), summary=None) for r in records]
            channels['keyword'] = [r.id for r in self.store.ranker.rank(query, lexical, now=as_of)[:pool]]
            timings['keyword'] = time.perf_counter() - stage_started
        if 'semantic' in active_channels:
            stage_started = time.perf_counter()
            # Sync the whole current user's corpus, then filter results. A restrictive
            # session query must not evict other sessions from the persistent index.
            corpus = records if as_of or known_at else list({r.id: r for r in [*self.store.list(user_id=user_id), *records]}.values())
            vectors = self.vector_index(len(corpus))
            vectors.sync(corpus, user_id=user_id)
            for hit in vectors.search(query, user_id=user_id, allowed_ids=allowed, limit=pool * 3):
                dense.setdefault(hit['memory_id'], hit)
            vector_diagnostics = dict(getattr(vectors, 'last_search', {}) or {})
            channels['semantic'] = list(dense)[:pool]
            timings['vector'] = time.perf_counter() - stage_started
        if 'graph' in active_channels:
            stage_started = time.perf_counter()
            self.graph.sync(self.store._read_log(), user_id=user_id)
            desired_predicates = graph_predicates(query)
            seeds = self.graph.entities(query, user_id=user_id, known_at=known_at)
            if re.search(r'\b(my|me|i)\b', query, re.I):
                seeds += self.graph.entities('User', user_id=user_id, known_at=known_at, exact=True)
            paths = self.graph.traverse([s['id'] for s in seeds], user_id=user_id,
                                       as_of=as_of, known_at=known_at, max_hops=max_hops, allowed_ids=allowed)
            if mode == 'hybrid':
                paths = [path for path in paths if desired_predicates and
                         all(edge['predicate'] in desired_predicates for edge in path['edges'])]
                paths.sort(key=lambda path: (-sum(edge['predicate'] in desired_predicates
                                                   for edge in path['edges']),
                                             len(path['edges']), tuple(path['source_ids'])))
            # Negative assertions are useful direct evidence, but are never traversed
            # as positive links when constructing multi-hop paths.
            seed_ids = {s['id'] for s in seeds}
            negatives = [e['source_id'] for e in self.graph.relations(user_id=user_id, as_of=as_of, known_at=known_at)
                         if not e['positive'] and e['source_id'] in allowed
                         and (not desired_predicates or e['predicate'] in desired_predicates)
                         and (e['subject'] in seed_ids or e['object'] in seed_ids)]
            channels['graph'] = list(dict.fromkeys([*negatives, *(identifier for path in paths for identifier in path['source_ids'])]))[:pool]
            timings['graph'] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        weights = {'keyword': .3, 'semantic': .5, 'graph': .2, 'recency': 1.0}
        scores, details = {}, {}
        for channel, identifiers in channels.items():
            for rank, identifier in enumerate(identifiers, 1):
                contribution = (weights[channel] if mode == 'hybrid' else 1) / (60 + rank)
                scores[identifier] = scores.get(identifier, 0) + contribution
                details.setdefault(identifier, {})[channel] = {'rank': rank, 'contribution': contribution}
        # Reward agreement between independent channels. Benchmark failures often
        # contained a lexical hit that was also semantically supported but lost a
        # top-k slot to a one-channel candidate.
        if mode == 'hybrid':
            for identifier, signals in details.items():
                scores[identifier] += .002 * max(0, len(signals) - 1)
        # Fuse at the old channel depth, then append deeper candidates for the reranker, so a
        # wider pool never reorders results when reranking is off or the model falls back.
        # fusion_limit lets an evaluator read a deep result list in exactly the order a call with
        # limit=fusion_limit produces (the depth otherwise follows the requested limit).
        depth = min(pool, max(20, 4 * (limit if fusion_limit is None else fusion_limit)))
        head = {}
        for identifier, signals in details.items():
            kept = [signal['contribution'] for signal in signals.values() if signal['rank'] <= depth]
            if kept:
                head[identifier] = sum(kept) + (.002 * (len(kept) - 1) if mode == 'hybrid' else 0)
        scores.update(head)
        fused = sorted(head, key=lambda key: (-head[key], key))
        fused += sorted((key for key in scores if key not in head), key=lambda key: (-scores[key], key))
        fused = fused[:pool]
        structured = bool(self.structure_steps) and should_rerank and mode == 'hybrid' and 'graph' in active_channels
        intent = time_intent(query)
        # Present-tense questions are about current validity; history cues or explicit
        # as_of/known_at timestamps switch off validity and version preferences.
        present = as_of is None and known_at is None
        validity = {'stale': {}, 'stale_predicates': {}, 'current': set(), 'cued': set()}
        injected = []
        if structured and fused and present and self.structure_steps & {'validity', 'versions'}:
            validity = {**self._validity(user_id, fused, allowed), 'cued': graph_predicates(query)}
            if 'validity' in self.structure_steps and intent != 'history':
                # A superseded statement's successor joins the reranked pool even when no channel
                # retrieved it, but only when the question asks about the superseded relation
                # ("Which city ...?" cues lives_in; "How much rent did I pay?" does not).
                pool_ids = set(fused)
                extra = [s for stale, successors in validity['stale'].items()
                         if validity['stale_predicates'][stale] & validity['cued']
                         for s in successors if s not in pool_ids]
                for identifier in dict.fromkeys(extra):
                    fused.append(identifier)
                    injected.append(identifier)
                    # A successor some channel found below the pool cut keeps its fusion evidence.
                    scores.setdefault(identifier, 0.0)
                    details.setdefault(identifier, {})
        snippets = {}
        for identifier in fused:
            record = by_id[identifier]
            if identifier in dense:
                snippet = dense[identifier]['text']
            else:
                terms = tokenize(query)
                record_chunks = chunks(record)
                snippet = max(record_chunks, key=lambda c: len(terms & tokenize(c.text))).text if record_chunks else ''
            snippets[identifier] = snippet
        timings['fusion'] = time.perf_counter() - stage_started

        reranker_scores = {}
        reranker_status = {'status': 'disabled', 'model': None, 'error_type': None}
        ordered = fused
        if should_rerank and fused:
            stage_started = time.perf_counter()
            try:
                model = self.reranker
                if model is None:
                    raise ValueError('Reranker is disabled')
                documents, owners = [], []
                terms = tokenize(query)
                for identifier in fused:
                    record = by_id[identifier]
                    choices = []
                    if identifier in dense:
                        choices.append(dense[identifier]['text'])
                    ranked_chunks = sorted(chunks(record),
                                           key=lambda item: (-len(terms & tokenize(item.text)), item.start))
                    choices.extend(item.text for item in ranked_chunks)
                    for text in list(dict.fromkeys(choices))[:3]:
                        documents.append(text)
                        owners.append((identifier, text))
                values = list(model.rerank(query, documents))
                if len(values) != len(documents) or any(not isinstance(value, (int, float))
                        or isinstance(value, bool) or not math.isfinite(value) for value in values):
                    raise ValueError('Reranker returned invalid scores')
                winning = {}
                for (identifier, text), value in zip(owners, values):
                    if identifier not in winning or value > winning[identifier][0]:
                        winning[identifier] = (float(value), text)
                for identifier, (value, text) in winning.items():
                    reranker_scores[identifier] = value
                    snippets[identifier] = text
                ordered = sorted(fused, key=lambda key: (-reranker_scores[key], -scores[key], key))
                reranker_status = {'status': 'applied', 'model': model.model_id, 'error_type': None}
            except Exception as exc:
                reranker_status = {'status': 'fallback', 'model': getattr(self._reranker, 'model_id', None),
                                   'error_type': type(exc).__name__}
            timings['reranker'] = time.perf_counter() - stage_started

        if injected and reranker_status['status'] != 'applied':
            # Without reranker scores there is no reordering, so nothing may enter that no channel found.
            fused = [identifier for identifier in fused if identifier not in injected]
            ordered = [identifier for identifier in ordered if identifier not in injected]
        adjustments = {}
        if structured and reranker_status['status'] == 'applied':
            ordered, adjustments = self._structured_order(
                query, ordered, reranker_scores, paths, by_id, validity, intent=intent, present=present)

        result = []
        timings['total'] = time.perf_counter() - search_started
        timing_ms = {key: value * 1000 for key, value in timings.items()}
        for identifier in ordered[:limit]:
            record = by_id[identifier]
            metadata = {**reranker_status, 'score': reranker_scores.get(identifier)}
            hit = {'memory_id': identifier, 'text': snippets[identifier], 'score': scores[identifier],
                   'fusion_score': scores[identifier], 'reranker_score': reranker_scores.get(identifier),
                   'cosine': dense[identifier]['score'] if identifier in dense else None,
                   'rank': len(result) + 1, 'reranker': metadata, 'candidate_pool_size': len(fused),
                   'channels': list(active_channels), 'timings_ms': timing_ms,
                   'vector_index': vector_diagnostics,
                   'signals': details[identifier], 'record': record.to_dict(),
                   'paths': [path for path in paths if identifier in path['source_ids']]}
            if structured and reranker_status['status'] == 'applied':
                hit['ordering'] = {'time_intent': intent, 'adjustments': adjustments.get(identifier, [])}
            result.append(hit)
        return result

    def _validity(self, user_id, identifiers, allowed):
        """Classify candidates by the validity of the graph edges they assert, as currently known.

        stale: every edge the memory asserts was closed before now; maps to the memories holding
        the currently valid edge with the same subject and predicate (its successors), choosing the
        edge whose start is closest to the stale edge's end. current: memories with a valid edge now.
        """
        now = iso(utc_now())
        known = 'known_from<=? AND (known_to IS NULL OR known_to>?)'
        placeholders = ','.join('?' for _ in identifiers)
        with self.database.connect() as db:
            rows = [dict(row) for row in db.execute(
                f'SELECT source_id,subject,predicate,positive,valid_from,valid_to FROM edges '
                f'WHERE user_id=? AND {known} AND source_id IN ({placeholders})',
                [user_id, now, now, *identifiers])]
            by_source = {}
            for row in rows:
                by_source.setdefault(row['source_id'], []).append(row)
            current = {source for source, edges in by_source.items()
                       if any(edge['valid_from'] <= now and (edge['valid_to'] is None or edge['valid_to'] > now)
                              for edge in edges)}
            stale, stale_predicates = {}, {}
            for source, edges in by_source.items():
                if any(edge['valid_to'] is None or edge['valid_to'] > now for edge in edges):
                    continue
                stale_predicates[source] = {edge['predicate'] for edge in edges}
                successors = []
                for edge in edges:
                    options = [dict(row) for row in db.execute(
                        f'SELECT source_id,valid_from FROM edges WHERE user_id=? AND subject=? AND predicate=? '
                        f'AND positive=? AND source_id<>? AND valid_from>=? AND valid_from<=? '
                        f'AND (valid_to IS NULL OR valid_to>?) AND {known}',
                        (user_id, edge['subject'], edge['predicate'], edge['positive'], source,
                         edge['valid_from'], now, now, now, now))]
                    options = [option for option in options if option['source_id'] in allowed]
                    if options:
                        end = datetime.fromisoformat(edge['valid_to'])
                        gap = lambda option: abs((datetime.fromisoformat(option['valid_from']) - end).total_seconds())
                        best = min(map(gap, options))
                        successors += sorted(option['source_id'] for option in options if gap(option) == best)
                stale[source] = list(dict.fromkeys(successors))
        return {'stale': stale, 'stale_predicates': stale_predicates, 'current': current}

    def _structured_order(self, query, ordered, reranker_scores, paths, by_id, validity, *, intent, present):
        """Reorder reranked candidates with graph structure and time, which the cross-encoder ignores.

        Each candidate is keyed by (graph tier, reranker score). "B takes A's place" gives B a key
        just above A's, so B moves directly ahead of A and nothing else changes position.
          graph:    relevant candidates that source a cued, valid graph path rank first.
          validity: a stale statement yields its place to the statement that superseded it.
          versions: for "now"/"latest" (or "originally"/"before") questions, the newer (older) of two
                    relevant statements by the same speaker about the same query topic goes first.
          chain:    the other memories of a relevant candidate's cued path follow it directly.
        """
        steps = self.structure_steps
        position = {identifier: index for index, identifier in enumerate(ordered)}
        top = max(reranker_scores[identifier] for identifier in ordered)
        relevant = {identifier for identifier in ordered
                    if reranker_scores[identifier] >= top - self.structure_band}
        path_index = {}
        for path in paths:
            for identifier in path['source_ids']:
                path_index.setdefault(identifier, []).append(path)
        tier = {identifier: int('graph' in steps and identifier in relevant and identifier in path_index)
                for identifier in ordered}
        key = {identifier: (tier[identifier], reranker_scores[identifier]) for identifier in ordered}
        notes = {identifier: (['graph_path'] if tier[identifier] else []) for identifier in ordered}

        def take_place(winner, loser, note):
            target = (key[loser][0], key[loser][1] + 1e-6)
            if key[winner] < target:
                key[winner] = target
                notes[winner].append(f'{note}:{loser}')
                return True
            return False

        if 'validity' in steps and present and intent != 'history':
            for identifier in sorted(validity['stale'], key=lambda item: key.get(item, (-1, 0)), reverse=True):
                # Only a relevant stale statement about a relation the question asks for yields its place.
                if (identifier in key and identifier in relevant
                        and validity['stale_predicates'].get(identifier, set()) & validity['cued']):
                    for successor in validity['stale'][identifier]:
                        if successor in key:
                            take_place(successor, identifier, 'supersedes')
        if 'versions' in steps and present and intent != 'neutral':
            newer_first = intent == 'current'
            note = 'newer_version_of' if newer_first else 'older_version_of'
            members = [identifier for identifier in ordered if identifier in relevant][:VERSION_MEMBERS]

            def speaker(identifier):
                value = by_id[identifier].source_metadata.get('speaker')
                return value.casefold() if isinstance(value, str) and value else None

            speakers = {identifier: speaker(identifier) for identifier in members}
            topics = {identifier: _topic_terms(by_id[identifier].content) for identifier in members}
            query_terms = _topic_terms(query)
            for _ in range(len(members)):
                changed = False
                for incumbent in sorted(members, key=key.get, reverse=True):
                    for challenger in members:
                        first, second = by_id[incumbent], by_id[challenger]
                        # The challenger must be the newer (current) or older (history) statement,
                        # by the same speaker, and currently ranked behind the incumbent.
                        if (incumbent == challenger or key[challenger] > key[incumbent]
                                or speakers[incumbent] != speakers[challenger]
                                or first.created_at == second.created_at
                                or (second.created_at > first.created_at) != newer_first):
                            continue
                        # Two statements that both assert currently valid graph relations are
                        # both true now; neither is a version of the other.
                        if incumbent in validity['current'] and challenger in validity['current']:
                            continue
                        # Same topic: both statements contain a content word of the question.
                        shared = (query_terms - tokenize(speakers[incumbent] or '')) & topics[incumbent] & topics[challenger]
                        if shared:
                            changed |= take_place(challenger, incumbent, note)
                if not changed:
                    break
        ranked = sorted(ordered, key=lambda identifier: (-key[identifier][0], -key[identifier][1], position[identifier]))
        if 'chain' in steps:
            chained, placed = [], set()
            for identifier in ranked:
                if identifier in placed:
                    continue
                chained.append(identifier)
                placed.add(identifier)
                if identifier not in relevant:
                    continue
                for path in sorted(path_index.get(identifier, []), key=lambda path: len(path['edges'])):
                    for partner in path['source_ids']:
                        if partner not in placed and partner in position:
                            chained.append(partner)
                            placed.add(partner)
                            notes[partner].append(f'chain_partner_of:{identifier}')
            ranked = chained
        return ranked, {identifier: values for identifier, values in notes.items() if values}

    @locked
    def context(self, query, *, user_id, budget=10000, **search_options):
        if not 256 <= budget <= 100000:
            raise ValueError('Context budget must be between 256 and 100000 characters')
        hits = self.search(query, user_id=user_id, **search_options)
        reranker = ({key: hits[0]['reranker'][key] for key in ('status', 'model', 'error_type')} if hits
                    else {'status': 'disabled', 'model': None, 'error_type': None})
        # Reranked evidence keeps the reranker's order, which the extractive answer shows
        # first. Fusion-only evidence prefers snippets that add query coverage. Both skip
        # near-duplicates so the budget is not spent twice on one statement.
        query_terms = tokenize(query)
        selected, covered = [], set()
        remaining = list(hits)
        reranked = reranker['status'] == 'applied'
        while remaining:
            best = remaining[0] if reranked else max(remaining, key=lambda hit: (
                len((tokenize(hit['text']) & query_terms) - covered),
                -max((len(tokenize(hit['text']) & tokenize(old['text'])) /
                      max(1, len(tokenize(hit['text']) | tokenize(old['text']))) for old in selected), default=0),
                -hit['rank']))
            remaining.remove(best)
            if any(len(tokenize(best['text']) & tokenize(old['text'])) /
                   max(1, len(tokenize(best['text']) | tokenize(old['text']))) > .9 for old in selected):
                continue
            selected.append(best)
            covered.update(tokenize(best['text']) & query_terms)
        hits = selected
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
        # Scores stay outside 'text', so they never reach a prompt as evidence.
        scores = {hit['memory_id']: {'rank': hit['rank'], 'reranker': hit['reranker_score'],
                                     'fusion': hit['fusion_score'], 'cosine': hit['cosine'],
                                     'channels': sorted(hit['signals'])}
                  for hit in hits if hit['memory_id'] in sources}
        return {'query': query, 'sources': sources, 'paths': paths, 'scores': scores, 'reranker': reranker,
                'text': '\n'.join(blocks), 'characters': len('\n'.join(blocks)),
                'as_of': iso(search_options.get('as_of') or utc_now()),
                'known_at': iso(search_options.get('known_at') or utc_now()),
                'retrieved': len(hits), 'historical': bool(search_options.get('as_of') or search_options.get('known_at'))}
