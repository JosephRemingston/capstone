"""Standard binary and graded retrieval metrics."""
from __future__ import annotations

import math


def retrieval_metrics(predicted, grades):
    predicted = list(dict.fromkeys(predicted))
    relevant = {key for key, value in grades.items() if value > 0}
    if not relevant:
        return {**{f'recall_at_{k}': None for k in (5, 10, 20)},
                **{f'hit_at_{k}': None for k in (5, 10)}, 'mrr': None,
                'ndcg_at_5': None, 'ndcg_at_10': None}
    result = {}
    for k in (5, 10, 20):
        result[f'recall_at_{k}'] = len(set(predicted[:k]) & relevant) / len(relevant)
    for k in (5, 10):
        result[f'hit_at_{k}'] = float(bool(set(predicted[:k]) & relevant))
    result['mrr'] = next((1 / rank for rank, key in enumerate(predicted, 1) if key in relevant), 0.0)
    ideal = sorted(grades.values(), reverse=True)
    for k in (5, 10):
        dcg = sum((2 ** grades.get(key, 0) - 1) / math.log2(rank + 1)
                  for rank, key in enumerate(predicted[:k], 1))
        idcg = sum((2 ** grade - 1) / math.log2(rank + 1)
                   for rank, grade in enumerate(ideal[:k], 1))
        result[f'ndcg_at_{k}'] = dcg / idcg if idcg else 0.0
    return result
