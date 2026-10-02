"""Validate and summarize completed blind ablation-review packets."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

DIMENSIONS = ('correctness', 'relevance', 'faithfulness', 'citation_usefulness', 'temporal_correctness')


def score(rows):
    if not rows:
        raise ValueError('No human reviews supplied')
    query_ids = {row.get('query_id') for row in rows}
    if not 50 <= len(query_ids) <= 100 or len(rows) != len(query_ids) * 4:
        raise ValueError('A complete packet needs four system reviews for each of 50–100 questions')
    if any({row['system_label'] for row in rows if row.get('query_id') == query_id}
           != {'System A', 'System B', 'System C', 'System D'} for query_id in query_ids):
        raise ValueError('Every question needs all four blind system labels')
    grouped, reviewers = defaultdict(lambda: defaultdict(list)), set()
    seen = set()
    for row in rows:
        reviewer = row.get('reviewer', '').strip()
        if not reviewer or not row.get('review_id') or (row['review_id'], reviewer) in seen:
            raise ValueError('Every review needs a reviewer and a unique review/reviewer pair')
        seen.add((row['review_id'], reviewer))
        reviewers.add(reviewer)
        try:
            reviewed = datetime.fromisoformat(row['reviewed_at'].replace('Z', '+00:00'))
        except (KeyError, ValueError, AttributeError) as exc:
            raise ValueError('Every review needs a valid reviewed_at timestamp') from exc
        if reviewed.tzinfo is None or reviewed > datetime.now(timezone.utc):
            raise ValueError('Review timestamps must be timezone-aware and nonfuture')
        ratings = row.get('scores')
        if not isinstance(ratings, dict) or set(ratings) != set(DIMENSIONS):
            raise ValueError('Every review must use the complete rubric')
        for dimension, value in ratings.items():
            if type(value) is not int or not 0 <= value <= 4:
                raise ValueError('Human ratings must be integers from 0 to 4')
            grouped[row['system_label']][dimension].append(value)
    return {'status': 'independently_reviewed', 'reviews': len(rows),
            'reviewers': sorted(reviewers),
            'means_0_to_4': {system: {dimension: statistics.mean(values)
                for dimension, values in ratings.items()} for system, ratings in grouped.items()}}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('reviews', type=Path)
    parser.add_argument('--output', type=Path, default=Path('docs/reports/ablation/human_scores.json'))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.reviews.read_text().splitlines() if line.strip()]
    result = score(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
