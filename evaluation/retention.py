"""LongMemEval evidence retention through the full memory pipeline.

    python -m evaluation.retention --download

Replays each oracle example's sessions through MemoryCore and store.ingest() with the
original timestamps, then checks which answer-bearing turns are still visible at the
question date. has_answer labels are used only for scoring. Oracle sessions omit
retrieval distractors, so this measures retention and cleanup safety, not QA accuracy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
from tempfile import TemporaryDirectory
import urllib.request
from unittest.mock import patch

from main.application.core import MemoryCore
from main.domain.models import MemoryInput
from main.storage.cleanup import cleanup
from main.storage.store import LocalMemoryStore

ROOT = Path('data/evaluation/longmemeval')
SOURCE = 'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned'
REVISION = '98d7416c24c778c2fee6e6f3006e7a073259d48f'
SHA256 = '821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c'


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, '%Y/%m/%d (%a) %H:%M').replace(tzinfo=timezone.utc)


def chronology_valid(row: dict) -> bool:
    asked = parse_date(row['question_date'])
    return all(parse_date(date) <= asked for date in row['haystack_dates'])


def replay(row: dict) -> dict:
    if not chronology_valid(row):
        raise ValueError('Sessions dated after the question cannot enter a historical replay')
    core, owner = MemoryCore(), row['question_id']
    asked = parse_date(row['question_date'])
    sessions = sorted(zip(row['haystack_dates'], row['haystack_session_ids'], row['haystack_sessions']),
                      key=lambda item: parse_date(item[0]))
    gold, ingested = set(), []
    with TemporaryDirectory() as directory:
        store = LocalMemoryStore(Path(directory) / 'records.jsonl')
        for date, session_id, turns in sessions:
            when = parse_date(date)
            for index, turn in enumerate(turns):
                identifier = f'{session_id}:{index}'
                if turn.get('has_answer'):
                    gold.add(identifier)
                if not turn['content'].strip():
                    continue
                with patch('main.domain.features.FeatureExtractor._now', return_value=when), \
                        patch('main.domain.lifecycle.utc_now', return_value=when), \
                        patch('main.storage.store.utc_now', return_value=when):
                    record = core.process(MemoryInput(turn['content'], owner, session_id, role=turn['role'], timestamp=when))
                    record.id = identifier
                    store.ingest(record)
                ingested.append(identifier)
        visible = store.list(user_id=owner, now=asked)
        retained = {record.id for record in visible} | {key for record in visible for key in record.evidence_ids}
        bytes_before = store.path.stat().st_size if store.path.exists() else 0
        cleaned = cleanup(store, user_id=owner, now=asked, apply=True)
        after = store.list(user_id=owner, now=asked)
        all_records = store.all()
    labelled = gold & set(ingested)
    return {'id': owner, 'type': row['question_type'], 'turns': len(ingested), 'gold_turns': len(gold),
            'keep_everything_retention': len(labelled) / len(gold) if gold else None,
            'memory_retention': len(gold & retained) / len(gold) if gold else None,
            'visible_fraction': len(visible) / len(ingested) if ingested else 0.0,
            'missing_gold_turns': sorted(gold - retained),
            'cleanup_preserved_visible_state': [r.to_dict() for r in after] == [r.to_dict() for r in visible],
            'cleanup_removed_memories': cleaned['removed_memories'],
            'bytes_before_cleanup': bytes_before, 'bytes_after_cleanup': cleaned['bytes_after'],
            'superseded': sum(r.memory_status == 'superseded' for r in all_records),
            'consolidated': sum(r.memory_status == 'consolidated' for r in all_records)}


def summarize(results: list[dict]) -> dict:
    summary = {}
    for kind in sorted({r['type'] for r in results}) + ['all']:
        selected = [r for r in results if kind == 'all' or r['type'] == kind]
        labelled = [r for r in selected if r['gold_turns']]
        summary[kind] = {'examples': len(selected), 'with_gold_turns': len(labelled),
                         'keep_everything_retention': statistics.fmean(r['keep_everything_retention'] for r in labelled) if labelled else None,
                         'memory_retention': statistics.fmean(r['memory_retention'] for r in labelled) if labelled else None,
                         'mean_visible_fraction': statistics.fmean(r['visible_fraction'] for r in selected) if selected else None}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--limit', type=int, default=None, help='debugging: replay only the first N examples')
    parser.add_argument('--output', type=Path, default=Path('docs/reports/longmemeval_retention.json'))
    args = parser.parse_args()
    if args.download:
        ROOT.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f'{SOURCE}/resolve/{REVISION}/longmemeval_oracle.json', timeout=120) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != SHA256:
            raise ValueError('LongMemEval checksum mismatch')
        (ROOT / 'oracle.json').write_bytes(payload)
    payload = (ROOT / 'oracle.json').read_bytes()
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError('LongMemEval checksum mismatch; run with --download')
    rows = json.loads(payload)[:args.limit]
    excluded = [r['question_id'] for r in rows if not chronology_valid(r)]
    results = [replay(r) for r in rows if chronology_valid(r)]
    report = {'dataset': SOURCE, 'revision': REVISION, 'sha256': SHA256,
              'protocol': ('All oracle examples; original timestamps; has_answer used only for scoring; no model is '
                           'fitted here, but the self-disclosure retention rule was designed against LongMemEval failures.'),
              'chronology_exclusions': {'count': len(excluded), 'question_ids': excluded},
              'summary': summarize(results),
              'cleanup': {'examples': len(results),
                          'visible_state_preserved': sum(r['cleanup_preserved_visible_state'] for r in results),
                          'removed_memories': sum(r['cleanup_removed_memories'] for r in results),
                          'bytes_before': sum(r['bytes_before_cleanup'] for r in results),
                          'bytes_after': sum(r['bytes_after_cleanup'] for r in results)},
              'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
