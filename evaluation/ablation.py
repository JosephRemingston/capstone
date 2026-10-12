"""Controlled retrieval and extractive-answer ablation on the Claude-authored benchmark.

    python -m evaluation.ablation                          # test split, exact search, pool 20
    python -m evaluation.ablation --split dev --output-dir docs/reports/ablation_dev
    python -m evaluation.ablation --candidate-limit 100    # wider reranker pool
    python -m evaluation.ablation --ann                    # approximate-search arm

Every arm uses the same corpus, embedder, reranker, budget, and answer generator; only
the channels and reranking differ. Retrieval metrics are averaged over answerable
queries; answer and answerability metrics cover every query in the split.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from main.domain.models import MemoryRecord
from main.retrieval.embeddings import FastEmbedder
from main.storage.store import LocalMemoryStore

from .metrics import (answerability, context_metrics, correctness, mean, percentiles, pool_metrics,
                      retrieval_metrics)
from .schema import DATASET_PATH, GOLD_USER, QUERY_TYPES, load_dataset, parse_time
from .stats import paired_comparison
from .systems import (ARMS, COMPARISONS, CachingEmbedder, Protocol, answer_extractive, build_rag,
                      protocol_record, resolve_pool, search_options)

RETRIEVAL_KEYS = ('recall_at_5', 'recall_at_10', 'recall_at_20', 'hit_at_5', 'hit_at_10', 'ndcg_at_5',
                  'ndcg_at_10', 'mrr', 'pool_hit', 'pool_recall')
ANSWER_KEYS = ('correct_grounded', 'correct_top1', 'correct_first', 'context_precision', 'context_recall')
COMPARE_KEYS = ('recall_at_5', 'mrr', 'ndcg_at_10', 'correct_grounded')


def load_store(memories: list[dict], path: Path, mode: str) -> LocalMemoryStore:
    store = LocalMemoryStore(path)
    records = [MemoryRecord.from_dict(memory) for memory in memories]
    if mode == 'save':
        store._append(records)  # one batched raw append: same validation as save(), no reconciliation
    elif mode == 'ingest':
        for record in records:
            store.ingest(record)
    else:
        raise ValueError(f'Unknown load mode {mode!r}')
    return store


def gold_bucket(row: dict) -> str:
    count = sum(grade > 0 for grade in row['graded_relevance'].values())
    return 'none' if count == 0 else '1' if count == 1 else '2-3' if count <= 3 else '4+'


def run_query(rag, row: dict, arm: str, protocol: Protocol) -> dict:
    as_of = parse_time(row['as_of']) if row.get('as_of') else None
    options = search_options(arm, as_of=as_of)
    pool = resolve_pool(protocol.top_k, protocol.candidate_limit, rerank=options['rerank'])
    started = time.perf_counter()
    # The deep list is read in the order a product call with limit=top_k produces.
    hits = rag.search(row['query'], user_id=GOLD_USER, limit=min(pool, 100), candidate_limit=pool,
                      fusion_limit=protocol.top_k, **options)
    search_ms = (time.perf_counter() - started) * 1000
    predicted = [hit['memory_id'] for hit in hits]
    context = rag.context(row['query'], user_id=GOLD_USER, budget=protocol.budget, limit=protocol.top_k,
                          candidate_limit=pool, **options)
    response = answer_extractive(context, protocol.abstain_threshold)
    sources = list(context['sources'])
    top = hits[0] if hits else None
    return {'query_id': row['query_id'], 'arm': arm, 'query_type': row['query_type'], 'split': row['split'],
            'persona_id': row['persona_id'], 'unanswerable': row['unanswerable'], 'gold_bucket': gold_bucket(row),
            'predicted': predicted[:20], 'pool_size': len(predicted), 'sources': sources,
            'abstain': response['abstain'], 'status': 'ok',
            **retrieval_metrics(predicted, row['graded_relevance']),
            **pool_metrics(predicted, row['graded_relevance']),
            **correctness(row, context['sources'], top, response['abstain']),
            **context_metrics(row, sources),
            'top_reranker_score': top['reranker_score'] if top else None,
            'top_fusion_score': top['fusion_score'] if top else None,
            'reranker_status': top['reranker']['status'] if top else 'empty',
            'vector_algorithm': (top.get('vector_index') or {}).get('algorithm') if top else None,
            'search_ms': search_ms, 'stage_ms': top['timings_ms'] if top else {}}


def summarize(rows: list[dict]) -> dict:
    answerable = [row for row in rows if not row['unanswerable']]
    summary = {key: mean(row[key] for row in answerable) for key in RETRIEVAL_KEYS}
    summary['reranker_recovery'] = (summary['hit_at_5'] / summary['pool_hit']
                                    if summary['hit_at_5'] is not None and summary['pool_hit'] else None)
    summary['correct_grounded'] = mean(row['correct_grounded'] for row in rows)
    summary['correct_grounded_answerable'] = mean(row['correct_grounded'] for row in answerable)
    summary['correct_top1_answerable'] = mean(row['correct_top1'] for row in answerable)
    summary['correct_first_answerable'] = mean(row['correct_first'] for row in answerable)
    summary['context_precision'] = mean(row['context_precision'] for row in answerable)
    summary['context_recall'] = mean(row['context_recall'] for row in answerable)
    summary['answerability'] = answerability(rows)
    summary['latency_ms'] = percentiles([row['search_ms'] for row in rows])
    # Which vector search actually ran (exact, lsh, exact_fallback): an ANN arm can fall back to exact.
    summary['vector_paths'] = dict(Counter(row.get('vector_algorithm') or 'none' for row in rows))
    summary['queries'], summary['answerable_queries'] = len(rows), len(answerable)
    return summary


def breakdown(rows: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    result = {}
    for group, members in sorted(groups.items()):
        answerable = [row for row in members if not row['unanswerable']]
        result[group] = {'queries': len(members),
                         'recall_at_5': mean(r['recall_at_5'] for r in answerable),
                         'recall_at_10': mean(r['recall_at_10'] for r in answerable),
                         'hit_at_5': mean(r['hit_at_5'] for r in answerable),
                         'mrr': mean(r['mrr'] for r in answerable),
                         'pool_hit': mean(r['pool_hit'] for r in answerable),
                         'correct_grounded': mean(r['correct_grounded'] for r in members),
                         'abstain_rate': mean(float(r['abstain']) for r in members)}
    return result


def compare(rows_by_arm: dict, arms: list[str]) -> dict:
    result = {}
    for name, left, right in COMPARISONS:
        if left not in arms or right not in arms:
            continue
        left_rows = {row['query_id']: row for row in rows_by_arm[left]}
        right_rows = {row['query_id']: row for row in rows_by_arm[right]}
        ids = sorted(left_rows)
        result[name] = {'left': left, 'right': right, **{
            key: paired_comparison([left_rows[i][key] for i in ids], [right_rows[i][key] for i in ids])
            for key in COMPARE_KEYS}}
    return result


def failures(rows: list[dict], memories: dict, queries: dict, limit: int = 15) -> list[dict]:
    missed = [row for row in rows if not row['unanswerable'] and row['hit_at_5'] == 0]
    return [{'query_id': row['query_id'], 'query': queries[row['query_id']]['query'], 'type': row['query_type'],
             'gold': list(queries[row['query_id']]['graded_relevance']), 'in_pool': bool(row['pool_hit']),
             'top3': [(key, memories[key]['content'][:90]) for key in row['predicted'][:3]]}
            for row in missed[:limit]]


def pct(value) -> str:
    return '—' if value is None else f'{100 * value:.1f}%'


def markdown(report: dict) -> str:
    arms = report['arms']
    lines = [f"# Controlled ablation ({report['split']} split, {report['queries']} queries)", '',
             f"Dataset: {report['dataset']['label_provenance']}. Exact search: {not report['protocol']['ann']}. "
             f"Candidate pool per arm: {report['protocol']['resolved_pool']}. Top-K for answers: {report['protocol']['top_k']}.", '',
             '## Retrieval (answerable queries)', '',
             '| Arm | Recall@5 | Recall@10 | Hit@5 | MRR | nDCG@10 | Pool hit | Reranker recovery | P50 ms | P95 ms |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in arms:
        s = report['summary'][arm]
        lines.append(f"| {arm} | {pct(s['recall_at_5'])} | {pct(s['recall_at_10'])} | {pct(s['hit_at_5'])} | "
                     f"{s['mrr']:.3f} | {s['ndcg_at_10']:.3f} | {pct(s['pool_hit'])} | {pct(s['reranker_recovery'])} | "
                     f"{s['latency_ms']['p50']:.0f} | {s['latency_ms']['p95']:.0f} |")
    lines += ['', '## Answers and answerability (all queries, extractive generator)', '',
              '| Arm | Correct (grounded) | Correct top-1 (answerable) | First statement correct (answerable) | '
              'Answerable accepted | Unanswerable refused | Balanced | Context precision |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in arms:
        s, a = report['summary'][arm], report['summary'][arm]['answerability']
        lines.append(f"| {arm} | {pct(s['correct_grounded'])} | {pct(s['correct_top1_answerable'])} | "
                     f"{pct(s.get('correct_first_answerable'))} | {pct(a['answerable_accept_rate'])} | {pct(a['unanswerable_refusal_rate'])} | "
                     f"{pct(a['balanced_accuracy'])} | {pct(s['context_precision'])} |")
    lines += ['', '## Recall@5 by query type', '', '| Type | ' + ' | '.join(arms) + ' |',
              '| --- |' + ' ---: |' * len(arms)]
    for kind in QUERY_TYPES:
        cells = [pct(report['by_query_type'][arm].get(kind, {}).get('recall_at_5')) for arm in arms]
        lines.append(f'| {kind} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Paired comparisons (right minus left)', '']
    for name, item in report['comparisons'].items():
        r5, cg = item['recall_at_5'], item['correct_grounded']
        lines.append(f"- {name} ({item['left']} → {item['right']}): Recall@5 {r5['mean_difference']:+.3f} "
                     f"[{r5['ci95'][0]:+.3f}, {r5['ci95'][1]:+.3f}], p={r5['p_value']:.4f}; "
                     f"grounded correctness {cg['mean_difference']:+.3f}, p={cg['p_value']:.4f}.")
    lines += ['', 'Labels are Claude-authored and not human-validated. Grounded correctness requires the answer '
              'to appear in a supporting memory that reached the context.']
    return '\n'.join(lines) + '\n'


def run(*, dataset: Path, split: str, protocol: Protocol, arms: list[str], limit_queries: int | None = None,
        embedder=None, reranker='auto') -> dict:
    payload, queries = load_dataset(dataset, split)
    queries = sorted(queries, key=lambda q: (q['query_type'] == 'historical', q['query_id']))
    if limit_queries:
        queries = queries[:limit_queries]
    memories = {m['id']: m for m in payload['memories']}
    embedder = CachingEmbedder(embedder or FastEmbedder(), [q['query'] for q in queries])
    rows_by_arm = {arm: [] for arm in arms}
    started = time.perf_counter()
    with TemporaryDirectory() as directory:
        store = load_store(payload['memories'], Path(directory) / 'benchmark.jsonl', protocol.load_mode)
        visible = len(store.list(user_id=GOLD_USER))
        rag = build_rag(store, embedder=embedder, protocol=protocol, reranker=reranker)
        rag.sync(user_id=GOLD_USER)
        for row in queries:
            for arm in arms:
                rows_by_arm[arm].append(run_query(rag, row, arm, protocol))
        record = protocol_record(protocol, embedder=embedder, rag=rag,
                                 resolved_pool={arm: resolve_pool(protocol.top_k, protocol.candidate_limit,
                                                                  rerank=ARMS[arm]['rerank']) for arm in arms})
    if protocol.load_mode == 'save' and visible != len(payload['memories']):
        raise RuntimeError(f'{len(payload["memories"]) - visible} gold memories were hidden after loading')
    text = Path(dataset).read_text()
    return {'schema_version': 1, 'created_at': datetime.now().astimezone().isoformat(), 'split': split,
            'queries': len(queries), 'arms': arms, 'protocol': record,
            'dataset': {'path': str(dataset), 'sha256': hashlib.sha256(text.encode()).hexdigest(),
                        'memories': len(payload['memories']), 'visible_after_load': visible,
                        'label_provenance': payload['label_provenance']},
            'summary': {arm: summarize(rows_by_arm[arm]) for arm in arms},
            'by_query_type': {arm: breakdown(rows_by_arm[arm], 'query_type') for arm in arms},
            'by_gold_bucket': {arm: breakdown(rows_by_arm[arm], 'gold_bucket') for arm in arms},
            'comparisons': compare(rows_by_arm, arms),
            'failure_cases': {arm: failures(rows_by_arm[arm], memories, {q['query_id']: q for q in queries}) for arm in arms},
            'elapsed_seconds': time.perf_counter() - started,
            'rows': [row for arm in arms for row in rows_by_arm[arm]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, default=DATASET_PATH)
    parser.add_argument('--split', choices=('dev', 'test', 'all'), default='test')
    parser.add_argument('--output-dir', type=Path, default=Path('docs/reports/ablation'))
    parser.add_argument('--candidate-limit', default='20', help="integer pool size, or 'default' for the shipped formula")
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--ann', action='store_true', help='use approximate vector search')
    parser.add_argument('--abstain-threshold', type=float, default=None)
    parser.add_argument('--load-mode', choices=('save', 'ingest'), default='save')
    parser.add_argument('--structure', default='default',
                        help="post-rerank reordering: 'default', 'none', or steps like graph,validity,chain,versions")
    parser.add_argument('--arms', default=','.join(ARMS))
    parser.add_argument('--limit-queries', type=int, default=None, help='debugging: evaluate only the first N queries')
    args = parser.parse_args()
    candidate_limit = None if args.candidate_limit == 'default' else int(args.candidate_limit)
    arms = [arm.strip() for arm in args.arms.split(',') if arm.strip()]
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit(f'Unknown arms: {sorted(unknown)}')
    protocol = Protocol(candidate_limit=candidate_limit, top_k=args.top_k, ann=args.ann,
                        abstain_threshold=args.abstain_threshold, load_mode=args.load_mode,
                        structure=args.structure)
    report = run(dataset=args.dataset, split=args.split, protocol=protocol, arms=arms, limit_queries=args.limit_queries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'results.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    (args.output_dir / 'results.md').write_text(markdown(report))
    print(markdown(report))


if __name__ == '__main__':
    main()
