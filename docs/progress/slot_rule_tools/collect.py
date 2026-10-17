"""Collect hybrid_full contexts (planned scores interface) for a benchmark split or the dev calibration set.

  cd repo && PYTHONPATH=. python ../tools/collect.py --set bench-dev --output ../out/ctx_bench_dev.json
  cd repo && PYTHONPATH=. python ../tools/collect.py --set calib --calib PATH --output ../out/ctx_calib.json
The rule under study is a pure function of the context, so contexts are collected once and replayed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from main.retrieval.embeddings import FastEmbedder

from evaluation.ablation import load_store
from evaluation.metrics import correctness
from evaluation.schema import DATASET_PATH, GOLD_USER, load_dataset, parse_time
from evaluation.systems import CachingEmbedder, Protocol, build_rag, resolve_pool, search_options


def calib_rows(path):
    rows = []
    for r in json.load(open(path)):
        rows.append({'query_id': r['id'], 'query': r['query'], 'query_type': r.get('form') or r.get('attribute_type'),
                     'persona_id': r['persona'], 'split': 'calib', 'unanswerable': not r['answerable'],
                     'graded_relevance': r['gold'], 'relevant_memory_ids': list(r['gold']),
                     'supporting_memory_ids': [k for k, v in r['gold'].items() if v >= 3],
                     'expected_answer': r['answer'], 'answer_aliases': r['aliases'],
                     'computed': bool(r.get('computed'))})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', required=True, choices=['bench-dev', 'bench-test', 'calib'])
    parser.add_argument('--calib', type=Path)
    parser.add_argument('--candidate-limit', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.set == 'calib':
        payload, _ = load_dataset(DATASET_PATH, 'dev')
        queries = calib_rows(args.calib)
    else:
        payload, queries = load_dataset(DATASET_PATH, args.set.split('-')[1])
    protocol = Protocol(candidate_limit=args.candidate_limit)
    pool = resolve_pool(protocol.top_k, protocol.candidate_limit)
    embedder = CachingEmbedder(FastEmbedder(), [q['query'] for q in queries])
    out = []
    with TemporaryDirectory(dir=Path(__file__).resolve().parent.parent / 'out') as directory:
        store = load_store(payload['memories'], Path(directory) / 'b.jsonl', 'save')
        rag = build_rag(store, embedder=embedder, protocol=protocol)
        rag.sync(user_id=GOLD_USER)
        for row in queries:
            as_of = parse_time(row['as_of']) if row.get('as_of') else None
            options = search_options('hybrid_full', as_of=as_of)
            hits = rag.search(row['query'], user_id=GOLD_USER, limit=min(pool, 100), candidate_limit=pool, **options)
            context = rag.context(row['query'], user_id=GOLD_USER, budget=protocol.budget, limit=protocol.top_k,
                                  candidate_limit=pool, **options)
            top = hits[0] if hits else None
            out.append({'set': args.set, 'query_id': row['query_id'], 'query': row['query'],
                        'query_type': row['query_type'], 'persona_id': row['persona_id'],
                        'unanswerable': row['unanswerable'], 'computed': row.get('computed', False),
                        'expected_answer': row['expected_answer'], 'aliases': row.get('answer_aliases', []),
                        'supporting': row['supporting_memory_ids'],
                        'gold_in_context': bool(set(row['graded_relevance']) & set(context['sources'])),
                        'context': {'query': context['query'], 'sources': context['sources'],
                                    'scores': context['scores'], 'reranker': context['reranker']},
                        **{k + '_noabstain': v for k, v in correctness(row, context['sources'], top, False).items()}})
    args.output.write_text(json.dumps(out, indent=1, default=str))
    print('wrote', len(out), 'rows to', args.output)


if __name__ == '__main__':
    main()
