"""Category, tier, and importance agreement on Claude-authored conversational cases.

    python -m evaluation.classification

Labels were written from the category/tier definitions before the classifier was run.
They are not human-validated; a score below 100% is a finding, not a test failure.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from main.application.core import MemoryCore
from main.domain.models import MemoryInput
from main.ensemble import describe_scorer

from .clock import pinned_clock
from .schema import DATASET_DIR

CASES = DATASET_DIR / 'conversational_cases.jsonl'
NOW = datetime(2026, 10, 12, 9, tzinfo=timezone.utc)


def load_cases(path: Path = CASES) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for row in rows:
        lo, hi = row['importance_range']
        if not 0 <= lo <= hi <= 1:
            raise ValueError(f'{row["id"]}: invalid importance range')
    return rows


def macro_f1(pairs: list[tuple[str, str]]) -> tuple[float, dict]:
    labels = sorted({label for pair in pairs for label in pair})
    per_class = {}
    for label in labels:
        tp = sum(e == label and p == label for e, p in pairs)
        fp = sum(e != label and p == label for e, p in pairs)
        fn = sum(e == label and p != label for e, p in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': tp + fn}
    return sum(item['f1'] for item in per_class.values()) / len(per_class), per_class


def range_baselines(results: list[dict]) -> dict:
    """Importance-range agreement of constant scores and per hand-labelled range, so a scorer's agreement can be read
    against what a constant achieves (40 of the 60 ranges are [.45, 1])."""
    ranges = [tuple(r['importance_range']) for r in results]
    agreement = lambda value: sum(lo <= value <= hi for lo, hi in ranges) / len(ranges)
    grid = [round(0.01 * i, 2) for i in range(101)]
    best = max(grid, key=agreement)
    groups = {}
    for r in results:
        key = '{:.2f}-{:.2f}'.format(*r['importance_range'])
        groups.setdefault(key, []).append(r['importance_in_range'])
    return {'constant': {str(v): agreement(v) for v in (0.1, 0.3, 0.5, 0.7, 0.9)},
            'best_constant': {'score': best, 'agreement': agreement(best)},
            'by_range': {key: {'cases': len(hits), 'agreement': sum(hits) / len(hits)} for key, hits in sorted(groups.items())}}


def evaluate(rows: list[dict]) -> dict:
    core = MemoryCore()
    results = []
    for row in rows:
        with pinned_clock(NOW):   # the heuristic's recency feature reads the clock; scoring happens "at" NOW
            record = core.process(MemoryInput(row['content'], 'classification-eval', row['id'], timestamp=NOW))
        lo, hi = row['importance_range']
        results.append({'id': row['id'], 'content': row['content'],
                        'expected_category': row['expected_category'], 'predicted_category': record.category.value,
                        'expected_tier': row['expected_tier'], 'predicted_tier': record.tier.value,
                        'importance_score': record.importance_score, 'importance_range': [lo, hi],
                        'category_correct': record.category.value == row['expected_category'],
                        'tier_correct': record.tier.value == row['expected_tier'],
                        'importance_in_range': lo <= record.importance_score <= hi})
    pairs = [(r['expected_category'], r['predicted_category']) for r in results]
    macro, per_class = macro_f1(pairs)
    confusion = Counter(f'{e} -> {p}' for e, p in pairs)
    n = len(results)
    return {'cases': n, 'label_source': 'claude_authored, labelled before running the classifier; not human-validated',
            'importance_model': describe_scorer(core.importance_scorer), 'clock': NOW.isoformat(),
            'importance_range_baselines': range_baselines(results),
            'category_accuracy': sum(r['category_correct'] for r in results) / n,
            'category_macro_f1': macro, 'per_class': per_class,
            'tier_accuracy': sum(r['tier_correct'] for r in results) / n,
            'importance_range_agreement': sum(r['importance_in_range'] for r in results) / n,
            'confusion': dict(sorted(confusion.items())),
            'errors': [r for r in results if not (r['category_correct'] and r['tier_correct'])],
            'results': results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cases', type=Path, default=CASES)
    parser.add_argument('--output', type=Path, default=Path('docs/reports/conversational_evaluation.json'))
    args = parser.parse_args()
    report = evaluate(load_cases(args.cases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('cases', 'category_accuracy', 'category_macro_f1', 'tier_accuracy',
                                            'importance_range_agreement')}, indent=2))


if __name__ == '__main__':
    main()
