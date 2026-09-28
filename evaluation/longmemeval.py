"""External knowledge-update/preference evidence retention through the memory pipeline.

Oracle evidence sessions are supplied by this dataset: this is NOT a full-corpus
retrieval benchmark, answer accuracy, or a conflict-decision accuracy score.
"""
import argparse
from collections import Counter, defaultdict
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
from main.storage.store import LocalMemoryStore
from main.storage.cleanup import cleanup

ROOT = Path('data/evaluation/longmemeval')
REVISION = '98d7416c24c778c2fee6e6f3006e7a073259d48f'
SHA256 = '821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c'
SOURCE = 'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned'


def parse_date(value):
    return datetime.strptime(value, '%Y/%m/%d (%a) %H:%M').replace(tzinfo=timezone.utc)


def chronology_valid(row):
    question_time = parse_date(row['question_date'])
    return all(parse_date(date) <= question_time for date in row['haystack_dates'])


def evaluate(row):
    if not chronology_valid(row):
        raise ValueError('Future sessions must not enter a historical evaluation')
    core = MemoryCore()
    owner = row['question_id']
    sessions = sorted(zip(row['haystack_dates'], row['haystack_session_ids'], row['haystack_sessions']),
                      key=lambda item: parse_date(item[0]))
    gold, input_ids, input_bytes = set(), [], 0
    with TemporaryDirectory() as temporary:
        store = LocalMemoryStore(Path(temporary) / 'records.jsonl')
        for date, session_id, turns in sessions:
            for index, turn in enumerate(turns):
                identifier = f'{session_id}:{index}'
                if turn.get('has_answer'):
                    gold.add(identifier)  # Scoring labels never enter metadata or text.
                text = turn['content']
                if not text.strip():
                    continue
                when = parse_date(date)
                # Replay ingestion at observation time, rather than penalizing every
                # historical input as if it arrived years late during evaluation.
                with patch('main.domain.features.FeatureExtractor._now', return_value=when), \
                     patch('main.domain.lifecycle.utc_now', return_value=when), \
                     patch('main.storage.store.utc_now', return_value=when):
                    record = core.process(MemoryInput(text, owner, session_id, role=turn['role'], timestamp=when))
                    record.id = identifier
                    store.ingest(record)
                input_ids.append(identifier)
                input_bytes += len(text.encode())
        all_records = store.all()
        visible = store.list(user_id=owner, now=parse_date(row['question_date']))
        retained = {r.id for r in visible}
        for record in visible:
            retained.update(record.evidence_ids)
        raw_gold = len(gold & set(input_ids)) / len(gold) if gold else None
        retained_gold = len(gold & retained) / len(gold) if gold else None
        before_bytes = store.path.stat().st_size if store.path.exists() else 0
        cleanup_result = cleanup(store, user_id=owner, now=parse_date(row['question_date']), apply=True)
        after = store.list(user_id=owner, now=parse_date(row['question_date']))
        preserved = [r.to_dict() for r in after] == [r.to_dict() for r in visible]
        return {'cleanup_removed_memories': cleanup_result['removed_memories'],
                'cleanup_bytes_before': before_bytes, 'cleanup_bytes_after': cleanup_result['bytes_after'],
                'cleanup_preserved_visible_memories': preserved,
                'id': owner, 'type': row['question_type'], 'turns': len(input_ids),
                'gold_turns': len(gold), 'raw_retention_baseline': raw_gold,
                'memory_gold_evidence_retention': retained_gold,
                'visible_roots': len(visible), 'visible_fraction': len(visible) / len(input_ids) if input_ids else 0,
                'visible_text_bytes': sum(len(r.content.encode()) for r in visible), 'input_text_bytes': input_bytes,
                'history_bytes': before_bytes,
                'superseded': sum(r.memory_status == 'superseded' for r in all_records),
                'consolidated': sum(r.memory_status == 'consolidated' for r in all_records),
                'missing_gold_turns': sorted(gold - retained)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--all-types', action='store_true', help='Evaluate all 500 examples, including 392 outside the earlier subset.')
    parser.add_argument('--output', type=Path, default=Path('docs/reports/longmemeval_retention.json'))
    args = parser.parse_args()
    if args.download:
        ROOT.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f'{SOURCE}/resolve/{REVISION}/longmemeval_oracle.json', timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != SHA256:
            raise ValueError('Dataset checksum mismatch')
        (ROOT / 'oracle.json').write_bytes(payload)
    payload = (ROOT / 'oracle.json').read_bytes()
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError('Dataset checksum mismatch')
    rows = json.loads(payload)
    requested = [row for row in rows if args.all_types or row['question_type'] in {'knowledge-update', 'single-session-preference'}]
    excluded = [row for row in requested if not chronology_valid(row)]
    results = [evaluate(row) for row in requested if chronology_valid(row)]
    summary = {}
    for kind in sorted({r['type'] for r in results}):
        selected = [r for r in results if r['type'] == kind]
        labeled = [r for r in selected if r['gold_turns']]
        summary[kind] = {'examples': len(selected), 'with_evidence_labels': len(labeled),
            'raw_retention_baseline': statistics.mean(r['raw_retention_baseline'] for r in labeled) if labeled else None,
            'memory_gold_evidence_retention': statistics.mean(r['memory_gold_evidence_retention'] for r in labeled) if labeled else None,
            'mean_visible_fraction': statistics.mean(r['visible_fraction'] for r in selected),
            'superseded_memories': sum(r['superseded'] for r in selected),
            'consolidated_memories': sum(r['consolidated'] for r in selected)}
    report = {'dataset': SOURCE, 'revision': REVISION, 'sha256': SHA256,
        'protocol': 'All oracle examples' + (' (500)' if args.all_types else ' in the update/preference subset') + '; no fitting; timestamps respected; has_answer labels used only for scoring.',
        'chronology_exclusions': {'count': len(excluded), 'question_ids': [r['question_id'] for r in excluded],
                                 'reason': 'Evidence session dated after the question; excluded before scoring.'},
        'previously_evaluated_subset': '108 update/preference examples; remaining 392 are new to this evaluation run when --all-types is used.',
        'limits': 'Evidence retention and storage efficiency, not QA accuracy or correctness of individual conflict decisions. Oracle sessions remove retrieval distractors. Generic preference advice may require generation and human judgment. Ingestion clocks replay session timestamps; visibility is measured at each question date.',
        'cleanup': {'examples': len(results),
                    'visible_state_preserved': sum(r['cleanup_preserved_visible_memories'] for r in results),
                    'removed_memories': sum(r['cleanup_removed_memories'] for r in results),
                    'bytes_before': sum(r['cleanup_bytes_before'] for r in results),
                    'bytes_after': sum(r['cleanup_bytes_after'] for r in results)},
        'summary': summary, 'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

