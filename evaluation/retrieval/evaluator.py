"""Evaluate all four controlled retrieval systems."""
from __future__ import annotations

import time
from datetime import datetime

from .baselines import SYSTEMS, retrieve
from .metrics import retrieval_metrics


def evaluate_query(rag, row, *, user_id='gold-user', candidate_limit=20):
    results = []
    filters = {key: datetime.fromisoformat(row[key]) for key in ('as_of', 'known_at') if row.get(key)}
    for system in SYSTEMS:
        system_rag = rag[system] if isinstance(rag, dict) else rag
        started = time.perf_counter()
        hits = retrieve(system_rag, row['query'], system, user_id=user_id, limit=20,
                        candidate_limit=candidate_limit, **filters)
        elapsed = (time.perf_counter() - started) * 1000
        predicted = [hit['memory_id'] for hit in hits]
        timings = hits[0]['timings_ms'] if hits else {'total': elapsed}
        paths = [path for hit in hits for path in hit.get('paths', [])]
        support = set(row['supporting_memory_ids'])
        results.append({'query_id': row['query_id'], 'query_type': row['query_type'],
                        'system': system, 'predicted': predicted,
                        'candidate_pool_size': hits[0]['candidate_pool_size'] if hits else 0,
                        'reranker_status': hits[0]['reranker']['status'] if hits else 'empty',
                        'graph_path_found': (float(any(support <= set(path['source_ids']) for path in paths))
                                             if row['query_type'] == 'multi_hop' else None),
                        'latency_ms': elapsed, 'stage_latency_ms': timings,
                        **retrieval_metrics(predicted, row['graded_relevance'])})
    return results
