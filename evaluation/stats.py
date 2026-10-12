"""Paired statistics for comparing systems on the same queries."""
from __future__ import annotations

import random
import statistics


def paired_comparison(left: list, right: list, *, resamples: int = 5000, seed: int = 42) -> dict:
    """Mean difference (right - left), paired-bootstrap 95% CI, and sign-flip randomization p.

    Pairs where either value is None (for example unanswerable rows for recall) are dropped.
    """
    if len(left) != len(right):
        raise ValueError('Paired comparison needs aligned lists')
    differences = [b - a for a, b in zip(left, right) if a is not None and b is not None]
    if not differences:
        return {'pairs': 0, 'mean_difference': None, 'ci95': [None, None], 'p_value': None}
    rng = random.Random(seed)
    observed = statistics.fmean(differences)
    boot = sorted(statistics.fmean(rng.choices(differences, k=len(differences))) for _ in range(resamples))
    low, high = boot[int(0.025 * resamples)], boot[int(0.975 * resamples) - 1]
    extreme = 0
    for _ in range(resamples):
        flipped = statistics.fmean(value if rng.random() < 0.5 else -value for value in differences)
        extreme += abs(flipped) >= abs(observed) - 1e-12
    return {'pairs': len(differences), 'mean_difference': observed, 'ci95': [low, high],
            'p_value': (extreme + 1) / (resamples + 1)}


def cluster_bootstrap(groups: dict[str, list[float]], *, resamples: int = 1000, seed: int = 42) -> list[float]:
    """95% interval of the pooled mean when resampling whole groups (e.g. conversations)."""
    values = [group for group in groups.values() if group]
    if not values:
        return [None, None]
    rng = random.Random(seed)
    boot = sorted(statistics.fmean([item for group in rng.choices(values, k=len(values)) for item in group])
                  for _ in range(resamples))
    return [boot[int(0.025 * resamples)], boot[int(0.975 * resamples) - 1]]
