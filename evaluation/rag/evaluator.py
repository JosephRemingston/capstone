"""Answer and context evaluation using one fixed generation configuration."""
from __future__ import annotations

import re
import time
from datetime import datetime

from main.retrieval.generation import answer
from evaluation.retrieval.baselines import SYSTEMS
from .citation_eval import evaluate_citations


def _normalized(text):
    return ' '.join(re.findall(r'[a-z0-9]+', (text or '').casefold()))


def evaluate_query(rag, row, *, generator='extractive', client=None, user_id='gold-user',
                   top_k=5, candidate_limit=20, budget=10000):
    output = []
    filters = {key: datetime.fromisoformat(row[key]) for key in ('as_of', 'known_at') if row.get(key)}
    for system, options in SYSTEMS.items():
        system_rag = rag[system] if isinstance(rag, dict) else rag
        started = time.perf_counter()
        context_started = time.perf_counter()
        context = system_rag.context(row['query'], user_id=user_id, mode='hybrid', limit=top_k,
                              candidate_limit=candidate_limit, channels=options['channels'],
                              rerank=options['rerank'], budget=budget, **filters)
        context_latency = (time.perf_counter() - context_started) * 1000
        generation_started = time.perf_counter()
        response = answer(context, generator=generator, client=client)
        generation_latency = (time.perf_counter() - generation_started) * 1000
        retrieved = set(context['sources'])
        relevant = set(row['relevant_memory_ids'])
        supporting = set(row['supporting_memory_ids'])
        expected = _normalized(row.get('expected_answer'))
        actual = _normalized(response['answer'])
        unanswerable = row['unanswerable']
        correctness = float(response['abstain']) if unanswerable else float(bool(expected) and expected in actual)
        context_precision = len(retrieved & relevant) / len(retrieved) if retrieved else float(unanswerable)
        context_recall = len(retrieved & relevant) / len(relevant) if relevant else float(not retrieved)
        citations = evaluate_citations(response, context, supporting)
        unsupported = 0
        for statement in response.get('statements', []):
            if not statement.get('evidence'):
                unsupported += 1
        statement_count = len(response.get('statements', []))
        faithfulness = citations['citation_quote_validity']
        output.append({'query_id': row['query_id'], 'query_type': row['query_type'],
                       'system': system, 'answer': response['answer'], 'abstain': response['abstain'],
                       'retrieved': list(context['sources']), 'correctness': correctness,
                       'answer_relevance': correctness, 'faithfulness': faithfulness,
                       'context_precision': context_precision, 'context_recall': context_recall,
                       'unsupported_claim_rate': unsupported / statement_count if statement_count else 0.0,
                       'abstention_accuracy': float(response['abstain'] == unanswerable),
                       'context_latency_ms': context_latency, 'generation_latency_ms': generation_latency,
                       'end_to_end_latency_ms': (time.perf_counter() - started) * 1000,
                       **citations})
    return output
