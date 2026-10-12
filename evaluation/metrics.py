"""Retrieval, pool, answer, and answerability metrics. Pure functions over identifiers and text."""
from __future__ import annotations

import math
import re
import statistics


def normalize(text: str) -> str:
    """Lowercase alphanumeric tokens joined by single spaces."""
    return ' '.join(re.findall(r'[a-z0-9]+', text.casefold()))


def contains_phrase(text: str, phrase: str) -> bool:
    """Whole-token phrase containment after normalization."""
    needle = normalize(phrase)
    return bool(needle) and f' {needle} ' in f' {normalize(text)} '


def answer_forms(row: dict) -> list[str]:
    forms = [row['expected_answer'], *row.get('answer_aliases', [])] if row.get('expected_answer') else []
    return [form for form in forms if normalize(form)]


def relevant_ids(grades: dict) -> set[str]:
    return {key for key, grade in grades.items() if grade > 0}


def retrieval_metrics(predicted: list[str], grades: dict, ks=(5, 10, 20)) -> dict:
    """Recall/Hit/nDCG at each K plus MRR. All values are None for queries without evidence."""
    predicted = list(dict.fromkeys(predicted))
    relevant = relevant_ids(grades)
    names = [f'{name}_at_{k}' for k in ks for name in ('recall', 'hit', 'ndcg')] + ['mrr']
    if not relevant:
        return dict.fromkeys(names)
    result = {}
    ideal = sorted(grades.values(), reverse=True)
    for k in ks:
        top = predicted[:k]
        found = len(set(top) & relevant)
        result[f'recall_at_{k}'] = found / len(relevant)
        result[f'hit_at_{k}'] = float(found > 0)
        dcg = sum((2 ** grades.get(key, 0) - 1) / math.log2(rank + 1) for rank, key in enumerate(top, 1))
        idcg = sum((2 ** grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal[:k], 1))
        result[f'ndcg_at_{k}'] = dcg / idcg if idcg else 0.0
    result['mrr'] = next((1 / rank for rank, key in enumerate(predicted, 1) if key in relevant), 0.0)
    return result


def pool_metrics(pool_ids: list[str], grades: dict) -> dict:
    """Whether the gold evidence reached the candidate pool handed to the reranker."""
    relevant = relevant_ids(grades)
    if not relevant:
        return {'pool_hit': None, 'pool_recall': None}
    found = len(set(pool_ids) & relevant)
    return {'pool_hit': float(found > 0), 'pool_recall': found / len(relevant)}


def correctness(row: dict, sources: dict, top_hit: dict | None, abstained: bool) -> dict:
    """Grounded correctness (primary) and strict top-1 correctness.

    grounded: the answer did not abstain and an accepted answer form appears in a
    supporting memory that reached the context. A different persona's snippet that
    happens to contain the same string does not count.
    top1: the first-ranked retrieval hit is a supporting memory containing an answer form.
    first: the first context source, the statement an extractive answer shows first, is.
    Unanswerable rows are correct exactly when the system abstained.
    """
    if row['unanswerable']:
        return {'correct_grounded': float(abstained), 'correct_top1': float(abstained),
                'correct_first': float(abstained)}
    forms, support = answer_forms(row), set(row['supporting_memory_ids'])
    grounded = (not abstained) and any(
        contains_phrase(source['text'], form) for key, source in sources.items() if key in support for form in forms)
    top1 = (not abstained) and top_hit is not None and top_hit['memory_id'] in support and any(
        contains_phrase(top_hit['text'], form) for form in forms)
    first_key = next(iter(sources), None)
    first = (not abstained) and first_key in support and any(
        contains_phrase(sources[first_key]['text'], form) for form in forms)
    return {'correct_grounded': float(grounded), 'correct_top1': float(top1), 'correct_first': float(first)}


def context_metrics(row: dict, source_ids: list[str]) -> dict:
    relevant = relevant_ids(row['graded_relevance'])
    if not relevant:
        return {'context_precision': None, 'context_recall': None}
    found = len(set(source_ids) & relevant)
    return {'context_precision': found / len(source_ids) if source_ids else 0.0,
            'context_recall': found / len(relevant)}


def answerability(rows: list[dict]) -> dict:
    """Confusion matrix of answer/abstain against answerable/unanswerable."""
    usable = [row for row in rows if row.get('status', 'ok') == 'ok']
    answered_answerable = sum(not r['abstain'] and not r['unanswerable'] for r in usable)
    abstained_answerable = sum(r['abstain'] and not r['unanswerable'] for r in usable)
    answered_unanswerable = sum(not r['abstain'] and r['unanswerable'] for r in usable)
    abstained_unanswerable = sum(r['abstain'] and r['unanswerable'] for r in usable)
    answerable = answered_answerable + abstained_answerable
    unanswerable = answered_unanswerable + abstained_unanswerable
    accept = answered_answerable / answerable if answerable else None
    refuse = abstained_unanswerable / unanswerable if unanswerable else None
    balanced = (accept + refuse) / 2 if accept is not None and refuse is not None else None
    return {'answerable_accept_rate': accept, 'unanswerable_refusal_rate': refuse,
            'balanced_accuracy': balanced,
            'confusion': {'answered_answerable': answered_answerable, 'abstained_answerable': abstained_answerable,
                          'answered_unanswerable': answered_unanswerable,
                          'abstained_unanswerable': abstained_unanswerable},
            'generation_failures': len(rows) - len(usable)}


def mean(values) -> float | None:
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def percentiles(values, points=(50, 95, 99)) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {f'p{point}': None for point in points}
    result = {}
    for point in points:
        position = (len(ordered) - 1) * point / 100
        low, high = math.floor(position), math.ceil(position)
        result[f'p{point}'] = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return result
