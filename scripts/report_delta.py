"""Print per-metric deltas between two tagged measurement snapshots.

    python scripts/report_delta.py baseline e2 --match locomo_retrieval.hybrid_reranked
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(tag: str) -> dict:
    path = ROOT / 'evaluation' / 'reports' / f'{tag}_summary.json'
    if not path.exists():
        raise SystemExit(f'No snapshot summary for tag {tag!r} at {path}')
    return json.loads(path.read_text())


# Settings and provenance, not metrics: never compared numerically. Cleanup counts (._cleanup.) and diagnostics are.
NOT_METRICS = ('._protocol.', '._vector_paths.')


def is_metric(key: str) -> bool:
    return not key.startswith('_scorers.') and not any(marker in key for marker in NOT_METRICS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('before')
    parser.add_argument('after')
    parser.add_argument('--match', action='append', default=[], help='only keys containing this text (repeatable)')
    parser.add_argument('--allow-dataset-mismatch', action='store_true',
                        help='compare snapshots built from different benchmark files (prints a warning)')
    args = parser.parse_args()
    first, second = load(args.before), load(args.after)
    datasets = [payload.get('manifest', {}).get('dataset_sha256') for payload in (first, second)]
    if datasets[0] != datasets[1]:
        message = f'dataset_sha256 differs: {args.before}={datasets[0]} {args.after}={datasets[1]}'
        if not args.allow_dataset_mismatch:
            raise SystemExit(message + ' (pass --allow-dataset-mismatch to compare anyway)')
        print(f'WARNING: {message}')
    before, after = first['flat'], second['flat']
    keys = sorted(k for k in set(before) & set(after)
                  if isinstance(before[k], (int, float)) and isinstance(after[k], (int, float))
                  and is_metric(k) and all(m in k for m in args.match))
    # Settings changes are always shown for every section whose metrics are printed, even under --match.
    sections = {k.split('.')[0] for k in keys} if args.match else None
    protocol = sorted(k for k in set(before) | set(after) if '._protocol.' in k and before.get(k) != after.get(k)
                      and (sections is None or k.split('.')[0] in sections))
    for key in protocol:
        print(f'protocol differs: {key}: {before.get(key)!r} -> {after.get(key)!r}')
    width = max((len(k) for k in keys), default=10)
    print(f"{'metric':<{width}}  {args.before:>10}  {args.after:>10}  {'delta':>10}")
    for key in keys:
        print(f'{key:<{width}}  {before[key]:>10.4f}  {after[key]:>10.4f}  {after[key] - before[key]:>+10.4f}')
    missing = sorted(k for k in set(before) ^ set(after) if is_metric(k) and all(m in k for m in args.match))
    if missing:
        print(f'{len(missing)} metrics exist in only one snapshot, e.g. {missing[:5]}')


if __name__ == '__main__':
    main()
