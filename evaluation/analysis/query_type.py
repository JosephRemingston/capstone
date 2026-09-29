"""Break metrics down by query type and system."""
from __future__ import annotations

from collections import defaultdict
import statistics


def summarize(rows, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['query_type'], row['system']].append(row)
    return {query_type: {system: {metric: statistics.mean(
                row[metric] for row in selected if row.get(metric) is not None)
                if any(row.get(metric) is not None for row in selected) else None
            for metric in metrics}
        for (kind, system), selected in grouped.items() if kind == query_type}
        for query_type in sorted({kind for kind, system in grouped})}
