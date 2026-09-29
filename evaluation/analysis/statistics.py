"""Deterministic descriptive and paired-bootstrap statistics."""
from __future__ import annotations

import math
import random
import statistics


def describe(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {'count': 0, 'mean': None, 'stddev': None, 'ci95': [None, None]}
    mean = statistics.mean(values)
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    margin = 1.96 * stddev / math.sqrt(len(values))
    return {'count': len(values), 'mean': mean, 'stddev': stddev,
            'ci95': [mean - margin, mean + margin]}


def percentiles(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {'p50': None, 'p95': None, 'p99': None}
    def pick(q):
        return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]
    return {'p50': pick(.50), 'p95': pick(.95), 'p99': pick(.99)}


def paired_comparison(rows, left, right, metric, *, samples=5000, seed=42):
    by_key = {(row['query_id'], row['system']): row for row in rows}
    pairs = [(by_key[key, right].get(metric), by_key[key, left].get(metric))
             for key in sorted({query for query, system in by_key if system == left})
             if (key, right) in by_key]
    differences = [a - b for a, b in pairs if a is not None and b is not None]
    if not differences:
        return {'count': 0, 'absolute_improvement': None, 'percentage_improvement': None,
                'ci95': [None, None], 'paired_randomization_p': None}
    observed = statistics.mean(differences)
    left_mean = statistics.mean(b for a, b in pairs if a is not None and b is not None)
    rng = random.Random(seed)
    bootstrap = [statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(samples)]
    bootstrap.sort()
    extreme = 0
    for _ in range(samples):
        randomized = statistics.mean(value * (-1 if rng.random() < .5 else 1) for value in differences)
        extreme += abs(randomized) >= abs(observed)
    return {'count': len(differences), 'absolute_improvement': observed,
            'percentage_improvement': observed / left_mean if left_mean else None,
            'ci95': [bootstrap[int(.025 * samples)], bootstrap[int(.975 * samples) - 1]],
            'paired_randomization_p': (extreme + 1) / (samples + 1)}
