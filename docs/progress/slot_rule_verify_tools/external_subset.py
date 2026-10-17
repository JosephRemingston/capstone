"""Collect hybrid_full contexts (pool 20, exact search) on EXTERNAL data, never used for tuning.
Adapted from design/verify-abstention/external.py: saves the full context so every rule is replayed on
identical retrieval. Stores/indexes are written only under this scratch directory.

  cd repo && PYTHONPATH=. python ../tools/external.py longmemeval --output ../heldout/ctx_lme.json
  cd repo && PYTHONPATH=. python ../tools/external.py locomo --output ../heldout/ctx_locomo.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import time

from main.domain.models import MemoryCategory, MemoryRecord, MemoryTier
from main.retrieval.embeddings import FastEmbedder

from evaluation.locomo import SnapshotStore, conversation_records
from evaluation.metrics import contains_phrase
from evaluation.systems import CachingEmbedder, Protocol, build_rag

SCRATCH = Path(__file__).resolve().parent.parent / 'out'
OPTS = dict(mode='hybrid', channels=('keyword', 'semantic', 'graph'), rerank=True)


def keep(ctx):
    return {'query': ctx['query'], 'sources': {k: {'text': v['text']} for k, v in ctx['sources'].items()},
            'scores': ctx['scores'], 'reranker': ctx['reranker']}


def grounded_hit(ctx, answer, gold):
    """Answer text appears in a gold-evidence source that reached the context (answerable rows only)."""
    return any(contains_phrase(v['text'], str(answer)) for k, v in ctx['sources'].items() if not gold or k in gold)


def lme_records(row):
    records, owner = [], 'u-' + row['question_id']
    for sid, date, session in zip(row['haystack_session_ids'], row['haystack_dates'], row['haystack_sessions']):
        when = datetime.strptime(date.split(' (')[0], '%Y/%m/%d').replace(tzinfo=timezone.utc)
        for t_index, turn in enumerate(session):
            if turn['role'] != 'user':
                continue
            records.append((MemoryRecord(id=f'{sid}:{t_index}', content=turn['content'], user_id=owner, session_id=str(sid),
                                         category=MemoryCategory.SEMANTIC, tier=MemoryTier.LONG_TERM,
                                         created_at=when, updated_at=when, recorded_at=when,
                                         source_metadata={'speaker': turn['role'], 'session_date': date}),
                            bool(turn.get('has_answer'))))
    return owner, records


def run_lme(args):
    data = json.load(open('data/evaluation/longmemeval/oracle.json'))
    base, out, started = FastEmbedder(), [], time.time()
    with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
        for i, row in enumerate(data[:int(__import__('os').environ.get('LME_N', '500'))]):
            owner, pairs = lme_records(row)
            gold = {r.id for r, has in pairs if has}
            store = SnapshotStore(Path(directory) / f'{i}.jsonl', [r for r, _ in pairs])
            rag = build_rag(store, embedder=CachingEmbedder(base, [row['question']]), protocol=Protocol(candidate_limit=20))
            rag.sync(user_id=owner)
            ctx = rag.context(row['question'], user_id=owner, budget=10000, limit=5, candidate_limit=20, **OPTS)
            unanswerable = row['question_id'].endswith('_abs')
            out.append({'set': 'longmemeval', 'query_id': row['question_id'], 'query_type': row['question_type'],
                        'query': row['question'], 'unanswerable': unanswerable, 'answer': str(row['answer']),
                        'gold_in_context': bool(gold & set(ctx['sources'])),
                        'answer_in_gold_context': None if unanswerable else grounded_hit(ctx, row['answer'], gold),
                        'context': keep(ctx)})
            if i % 50 == 0:
                print(i, round(time.time() - started), flush=True)
    return out


def run_locomo(args):
    conversations = json.load(open('data/evaluation/locomo/locomo10.json'))
    base, out = FastEmbedder(), []
    with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
        for row in conversations[:int(__import__('os').environ.get('LOCOMO_N', '10'))]:
            records, _ = conversation_records(row)
            owner = str(row['sample_id'])
            store = SnapshotStore(Path(directory) / f'{owner}.jsonl', records)
            rag = build_rag(store, embedder=CachingEmbedder(base, [qa['question'] for qa in row['qa']]),
                            protocol=Protocol(candidate_limit=20))
            rag.sync(user_id=owner)
            for index, qa in enumerate(row['qa']):
                ctx = rag.context(qa['question'], user_id=owner, budget=10000, limit=5, candidate_limit=20, **OPTS)
                gold = set(qa.get('evidence') or [])
                unanswerable = qa['category'] == 5
                answer = str(qa.get('answer', qa.get('adversarial_answer')))
                out.append({'set': 'locomo', 'query_id': f'{owner}:{index}', 'query_type': str(qa['category']),
                            'query': qa['question'], 'unanswerable': unanswerable, 'answer': answer,
                            'gold_in_context': bool(gold & set(ctx['sources'])),
                            'answer_in_gold_context': None if unanswerable else grounded_hit(ctx, answer, gold),
                            'context': keep(ctx)})
            print(owner, len(out), flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['longmemeval', 'locomo'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    SCRATCH.mkdir(exist_ok=True)
    out = run_lme(args) if args.dataset == 'longmemeval' else run_locomo(args)
    args.output.write_text(json.dumps(out, default=str))
    print('wrote', len(out), file=sys.stderr)


if __name__ == '__main__':
    main()
