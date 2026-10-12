"""External LoCoMo evidence-retrieval benchmark (raw turns only; never QA labels or summaries).

    python -m evaluation.locomo --download                    # exact search, pool 20 (saved protocol)
    python -m evaluation.locomo --candidate-limit 100         # wider reranker pool
    python -m evaluation.locomo --ann --output docs/reports/locomo_retrieval_ann.json

Measures whether the gold evidence turns are retrieved, not answer correctness.
Category 5 (adversarial) questions are reported separately as the external
"should refuse" set. The first three conversations in file order (conv-26, conv-30, conv-41)
were used to choose the reranker pool size; held_out_metrics covers the other seven.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

from main.domain.models import MemoryCategory, MemoryRecord, MemoryTier
from main.retrieval.embeddings import FastEmbedder
from main.storage.store import LocalMemoryStore

from .metrics import mean, percentiles, pool_metrics, retrieval_metrics
from .stats import cluster_bootstrap
from .systems import CachingEmbedder, Protocol, build_rag, protocol_record, resolve_pool

COMMIT = '3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376'
SHA256 = '79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4'
SOURCE = 'https://github.com/snap-research/locomo'
ROOT = Path('data/evaluation/locomo')
TUNING_CONVERSATIONS = 3   # used to choose the reranker pool size (see module docstring)
MODES = ('recency', 'keyword', 'semantic', 'graph', 'hybrid_fusion', 'hybrid_reranked')
CATEGORIES = {'1': 'multi-hop', '2': 'temporal', '3': 'open-domain', '4': 'single-hop', '5': 'adversarial'}
KEYS = ('recall_at_5', 'recall_at_10', 'hit_at_5', 'hit_at_10', 'mrr_at_5', 'ndcg_at_5', 'pool_hit', 'pool_recall')


def download(root: Path = ROOT) -> None:
    root.mkdir(parents=True, exist_ok=True)
    base = f'https://raw.githubusercontent.com/snap-research/locomo/{COMMIT}'
    for remote, local in (('data/locomo10.json', 'locomo10.json'), ('LICENSE.txt', 'LICENSE.txt')):
        request = urllib.request.Request(f'{base}/{remote}', headers={'User-Agent': 'CogniMem-evaluation'})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        if local.endswith('.json') and hashlib.sha256(payload).hexdigest() != SHA256:
            raise ValueError('LoCoMo checksum mismatch')
        (root / local).write_bytes(payload)


def conversation_records(row: dict) -> tuple[list[MemoryRecord], set[str]]:
    """Text turns become memories with speaker and session date; image-only turns are skipped."""
    records, image_only = [], set()
    conversation, owner = row['conversation'], str(row['sample_id'])
    for key in sorted(conversation):
        if not re.fullmatch(r'session_\d+', key):
            continue
        date = conversation[key + '_date_time']
        when = datetime.strptime(date, '%I:%M %p on %d %B, %Y').replace(tzinfo=timezone.utc)
        for turn in conversation[key]:
            if not isinstance(turn.get('text'), str) or not turn['text'].strip():
                image_only.add(turn['dia_id'])
                continue
            records.append(MemoryRecord(id=turn['dia_id'], content=turn['text'], user_id=owner, session_id=key,
                                        category=MemoryCategory.SEMANTIC, tier=MemoryTier.LONG_TERM,
                                        created_at=when, updated_at=when, recorded_at=when,
                                        source_metadata={'speaker': turn['speaker'], 'session_date': date}))
    return records, image_only


class SnapshotStore(LocalMemoryStore):
    """Read-only in-memory snapshot with the normal visibility and filtering semantics."""

    def __init__(self, path, records):
        super().__init__(path)
        self.records = records

    def all(self):
        return self.records

    def _read_log(self):
        return self.records


def score(predicted: list[str], gold: list[str], pool: list[str] | None) -> dict:
    grades = {key: 1 for key in gold}
    metrics = retrieval_metrics(predicted, grades, ks=(5, 10))
    metrics['mrr_at_5'] = next((1 / rank for rank, key in enumerate(predicted[:5], 1) if key in grades), 0.0)
    metrics.update(pool_metrics(pool, grades) if pool is not None else {'pool_hit': None, 'pool_recall': None})
    return metrics


def summarize(rows: list[dict]) -> dict:
    result = {key: mean(row[key] for row in rows) for key in KEYS}
    result['reranker_recovery'] = (result['hit_at_5'] / result['pool_hit']
                                   if result['pool_hit'] and result['hit_at_5'] is not None else None)
    groups = defaultdict(list)
    for row in rows:
        groups[row['conversation']].append(row['recall_at_5'])
    result['recall_at_5_ci95_conversation_bootstrap'] = cluster_bootstrap(groups)
    result['latency_ms'] = percentiles([row['latency_ms'] for row in rows])
    result['by_category'] = {CATEGORIES[c]: {'count': sum(r['category'] == c for r in rows),
                                             'recall_at_5': mean(r['recall_at_5'] for r in rows if r['category'] == c),
                                             'recall_at_10': mean(r['recall_at_10'] for r in rows if r['category'] == c),
                                             'hit_at_5': mean(r['hit_at_5'] for r in rows if r['category'] == c)}
                             for c in sorted({r['category'] for r in rows})}
    result['questions'] = len(rows)
    return result


def run(*, protocol: Protocol, modes=MODES, root: Path = ROOT, limit_conversations: int | None = None) -> dict:
    payload = (root / 'locomo10.json').read_bytes()
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError('LoCoMo checksum mismatch; run with --download')
    conversations = json.loads(payload)[:limit_conversations]
    base = FastEmbedder()
    pool = resolve_pool(protocol.top_k, protocol.candidate_limit)   # reranked pool, recorded in the protocol
    rows, excluded, record = [], Counter(), None
    started = time.perf_counter()
    for position, row in enumerate(conversations):
        split = 'tuning' if position < TUNING_CONVERSATIONS else 'held_out'
        records, _ = conversation_records(row)
        identifiers = {r.id for r in records}
        questions = []
        for index, qa in enumerate(row['qa']):
            gold = qa.get('evidence', [])
            if not gold:
                excluded['no_evidence'] += 1
            elif not all(key in identifiers for key in gold):
                excluded['evidence_not_in_text_turns'] += 1
            else:
                questions.append((index, qa))
        owner = str(row['sample_id'])
        embedder = CachingEmbedder(base, [qa['question'] for _, qa in questions])
        store = SnapshotStore(root / 'indexes' / f'{owner}.jsonl', records)
        rag = build_rag(store, embedder=embedder, protocol=protocol)
        rag.sync(user_id=owner)
        for index, qa in questions:
            for mode in modes:
                hybrid = mode.startswith('hybrid')
                mode_pool = resolve_pool(protocol.top_k, protocol.candidate_limit, rerank=mode == 'hybrid_reranked')
                call = time.perf_counter()
                hits = rag.search(qa['question'], user_id=owner, mode='hybrid' if hybrid else mode,
                                  limit=min(mode_pool, 100) if hybrid else 10, candidate_limit=mode_pool,
                                  fusion_limit=protocol.top_k if hybrid else None,
                                  rerank=mode == 'hybrid_reranked')
                latency = (time.perf_counter() - call) * 1000
                predicted = [hit['memory_id'] for hit in hits]
                rows.append({'conversation': owner, 'split': split, 'question_index': index, 'mode': mode,
                             'vector_algorithm': (hits[0].get('vector_index') or {}).get('algorithm') if hits else None,
                             'category': str(qa['category']), 'predicted': predicted[:10], 'latency_ms': latency,
                             'reranker_status': hits[0]['reranker']['status'] if hits else 'empty',
                             **score(predicted, qa['evidence'], predicted if hybrid else None)})
        record = record or protocol_record(protocol, embedder=embedder, rag=rag, resolved_pool={
            mode: resolve_pool(protocol.top_k, protocol.candidate_limit, rerank=mode == 'hybrid_reranked')
            for mode in modes if mode.startswith('hybrid')})
        print(json.dumps({'conversation': owner, 'questions': len(questions),
                          'elapsed_seconds': round(time.perf_counter() - started)}), flush=True)
    by_mode = {mode: [r for r in rows if r['mode'] == mode] for mode in modes}
    return {'dataset': 'LoCoMo', 'source': SOURCE, 'commit': COMMIT, 'sha256': SHA256,
            'indexed': 'text turns with speaker and session date only; no QA, observations, or summaries',
            'protocol': record, 'excluded_questions': dict(excluded),
            'scored_questions': len(rows) // max(1, len(modes)),
            'metrics': {mode: summarize(items) for mode, items in by_mode.items()},
            'answerable_metrics': {mode: summarize([r for r in items if r['category'] != '5'])
                                   for mode, items in by_mode.items()},
            'adversarial_metrics': {mode: summarize([r for r in items if r['category'] == '5'])
                                    for mode, items in by_mode.items() if any(r['category'] == '5' for r in items)},
            'held_out_metrics': {mode: summarize([r for r in items if r['split'] == 'held_out'])
                                 for mode, items in by_mode.items() if any(r['split'] == 'held_out' for r in items)},
            'vector_paths': {mode: dict(Counter(r['vector_algorithm'] or 'none' for r in items))
                             for mode, items in by_mode.items()},
            'elapsed_seconds': time.perf_counter() - started, 'results': rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--candidate-limit', default='20', help="integer pool size, or 'default' for the shipped formula")
    parser.add_argument('--ann', action='store_true')
    parser.add_argument('--structure', default='default',
                        help="post-rerank reordering: 'default', 'none', or steps like graph,validity,chain,versions")
    parser.add_argument('--modes', default=','.join(MODES))
    parser.add_argument('--limit-conversations', type=int, default=None)
    parser.add_argument('--output', type=Path, default=Path('docs/reports/locomo_retrieval.json'))
    args = parser.parse_args()
    if args.download:
        download()
    protocol = Protocol(candidate_limit=None if args.candidate_limit == 'default' else int(args.candidate_limit),
                        ann=args.ann, structure=args.structure)
    report = run(protocol=protocol, modes=[m.strip() for m in args.modes.split(',')],
                 limit_conversations=args.limit_conversations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for mode, item in report['metrics'].items():
        print(f"{mode:16s} R@5 {item['recall_at_5']:.3f}  R@10 {item['recall_at_10']:.3f}  Hit@5 {item['hit_at_5']:.3f}  "
              f"pool_hit {item['pool_hit'] if item['pool_hit'] is None else round(item['pool_hit'], 3)}  "
              f"p50 {item['latency_ms']['p50']:.0f} ms")


if __name__ == '__main__':
    main()
