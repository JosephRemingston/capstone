"""Run developer cases or independently supplied human-reviewed labels.

python -m evaluation.evaluate
python -m evaluation.evaluate --export-review evaluation/review_template.jsonl
python -m evaluation.evaluate --labels FILE --reviewed-only
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from main import MemoryCore, MemoryInput, MemoryCategory, MemoryTier

DEFAULT_LABELS = Path(__file__).with_name('conversational_cases.jsonl')


def evaluate(rows):
    core = MemoryCore()
    confusion = Counter()
    results = []
    for row in rows:
        expected_category, expected_tier = row['expected_category'], row['expected_tier']
        MemoryCategory(expected_category)
        MemoryTier(expected_tier)
        low, high = row['importance_range']
        if not 0 <= low <= high <= 1:
            raise ValueError('Invalid importance range')
        record = core.process(MemoryInput(content=row['content'], user_id='evaluation', session_id=row['id'],
                             timestamp=datetime.now(timezone.utc)))
        confusion[expected_category, record.category.value] += 1
        results.append({'id': row['id'], 'content': row['content'],
                        'label_source': row['label_source'],
                        'expected_category': expected_category, 'category': record.category.value,
                        'expected_tier': expected_tier, 'tier': record.tier.value,
                        'importance_range': [low, high], 'score': record.importance_score,
                        'category_correct': expected_category == record.category.value,
                        'tier_correct': expected_tier == record.tier.value,
                        'importance_in_range': low <= record.importance_score <= high})
    count = len(results)
    if not count:
        raise ValueError('No eligible labels; human review has not been supplied')
    f1s = []
    for category in sorted({r['expected_category'] for r in results} | {r['category'] for r in results}):
        tp = confusion[category, category]
        fp = sum(n for (a,b),n in confusion.items() if b == category and a != category)
        fn = sum(n for (a,b),n in confusion.items() if a == category and b != category)
        f1s.append(2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return {'count': count, 'label_sources': dict(Counter(r['label_source'] for r in rows)),
            'category_accuracy': sum(r['category_correct'] for r in results) / count,
            'category_macro_f1': sum(f1s) / len(f1s),
            'tier_accuracy': sum(r['tier_correct'] for r in results) / count,
            'importance_range_agreement': sum(r['importance_in_range'] for r in results) / count,
            'confusion': [{'expected': a, 'predicted': b, 'count': n} for (a,b),n in sorted(confusion.items())],
            'note': 'Range agreement measures a proposed policy, not importance regression accuracy. Developer cases are not independent human validation. No model was trained or thresholds fitted on these labels.',
            'results': results}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--labels', type=Path, default=DEFAULT_LABELS)
    parser.add_argument('--output', type=Path, default=Path('reports/conversational_evaluation.json'))
    parser.add_argument('--reviewed-only', action='store_true')
    parser.add_argument('--export-review', type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.labels.read_text().splitlines() if line.strip()]
    if args.export_review:
        review_rows = []
        for row in rows:
            review_rows.append({'id': row['id'], 'content': row['content'], 'label_source': 'pending_human_review',
                'expected_category': None, 'expected_tier': None, 'importance_range': None,
                'reviewer': '', 'reviewed_at': '', 'notes': '',
                'proposed_labels': {key: row[key] for key in ('expected_category', 'expected_tier', 'importance_range')}})
        args.export_review.write_text(''.join(json.dumps(row) + '\n' for row in review_rows))
        return
    if args.reviewed_only:
        rows = [row for row in rows if row.get('label_source') == 'human_reviewed' and row.get('reviewer') and row.get('reviewed_at')]
    report = evaluate(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('results', 'confusion')}, indent=2))
    for row in report['results']:
        if not all(row[key] for key in ('category_correct','tier_correct','importance_in_range')):
            print('Mismatch:', row['id'], row['content'], row['category'], row['tier'], row['score'])


if __name__ == '__main__':
    main()
