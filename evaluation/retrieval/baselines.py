"""Controlled retrieval systems for the graph/reranker ablation."""

SYSTEMS = {
    'vector': {'channels': ('semantic',), 'rerank': False},
    'vector_reranker': {'channels': ('semantic',), 'rerank': True},
    'vector_graph': {'channels': ('semantic', 'graph'), 'rerank': False},
    'vector_graph_reranker': {'channels': ('semantic', 'graph'), 'rerank': True},
}


def retrieve(rag, query, system, *, user_id, limit=20, candidate_limit=20, **time_filters):
    if system not in SYSTEMS:
        raise ValueError(f'Unknown ablation system: {system}')
    options = SYSTEMS[system]
    return rag.search(query, user_id=user_id, mode='hybrid', limit=limit,
                      candidate_limit=max(limit, candidate_limit), channels=options['channels'],
                      rerank=options['rerank'], **time_filters)
