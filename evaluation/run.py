"""Run the controlled CogniMem graph/reranker ablation.

Usage:
  .venv/bin/python -m evaluation.run
  .venv/bin/python -m evaluation.run --generator gemini --env-file .env
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import random
import statistics
from tempfile import TemporaryDirectory
import time

from main.domain.models import MemoryRecord
from main.retrieval.embeddings import FastEmbedder
from main.retrieval.generation import gemini_client
from main.retrieval.rag import MemoryRAG
from main.storage.store import LocalMemoryStore
from evaluation.analysis.plots import write_recall_chart
from evaluation.analysis.query_type import summarize as summarize_types
from evaluation.analysis.statistics import describe, paired_comparison, percentiles
from evaluation.datasets.build_gold import QUERY_TYPES
from evaluation.rag.evaluator import evaluate_query as evaluate_rag
from evaluation.retrieval.baselines import SYSTEMS
from evaluation.retrieval.evaluator import evaluate_query as evaluate_retrieval
from evaluation.quality import audit as quality_audit
from evaluation.lifecycle import evaluate as evaluate_lifecycle

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / 'datasets' / 'gold_queries.json'
RETRIEVAL_METRICS = ('recall_at_5', 'recall_at_10', 'recall_at_20', 'hit_at_5',
                     'hit_at_10', 'mrr', 'ndcg_at_5', 'ndcg_at_10')
RAG_METRICS = ('correctness', 'answer_relevance', 'faithfulness', 'context_precision',
               'context_recall', 'citation_precision', 'citation_recall',
               'citation_completeness', 'unsupported_claim_rate', 'abstention_accuracy')


def load_dataset(path=DEFAULT_DATASET):
    payload = json.loads(Path(path).read_text())
    queries, memories = payload.get('queries'), payload.get('memories')
    if not isinstance(queries, list) or not 200 <= len(queries) <= 500:
        raise ValueError('Gold dataset must contain 200–500 queries')
    if len({row.get('query_id') for row in queries}) != len(queries):
        raise ValueError('Every query_id must be unique')
    if set(QUERY_TYPES) - {row.get('query_type') for row in queries}:
        raise ValueError('Gold dataset does not cover every required query type')
    memory_ids = {row.get('id') for row in memories}
    required = {'query_id', 'query', 'query_type', 'relevant_memory_ids', 'graded_relevance',
                'expected_answer', 'supporting_memory_ids', 'unanswerable'}
    for row in queries:
        if not required <= set(row) or set(row['graded_relevance']) != set(row['relevant_memory_ids']):
            raise ValueError(f'Invalid query schema: {row.get("query_id")}')
        if any(type(grade) is not int or not 0 <= grade <= 3 for grade in row['graded_relevance'].values()):
            raise ValueError(f'Invalid relevance grade: {row["query_id"]}')
        if not set(row['relevant_memory_ids']) | set(row['supporting_memory_ids']) <= memory_ids:
            raise ValueError(f'Unknown memory reference: {row["query_id"]}')
    return payload


def summarize(rows, metrics):
    output = {}
    for system in SYSTEMS:
        selected = [row for row in rows if row['system'] == system]
        output[system] = {metric: describe(row.get(metric) for row in selected) for metric in metrics}
    return output


def latency_summary(retrieval, rag):
    output = {}
    for system in SYSTEMS:
        selected = [row for row in retrieval if row['system'] == system]
        answer_rows = [row for row in rag if row['system'] == system]
        stages = defaultdict(list)
        for row in selected:
            stages['retrieval'].append(row['latency_ms'])
            for key, value in row['stage_latency_ms'].items():
                stages[key].append(value)
        stages['context'].extend(row['context_latency_ms'] for row in answer_rows)
        stages['generation'].extend(row['generation_latency_ms'] for row in answer_rows)
        stages['end_to_end'].extend(row['end_to_end_latency_ms'] for row in answer_rows)
        output[system] = {stage: percentiles(values) for stage, values in stages.items()}
    return output


def make_human_packet(rag_rows, queries, *, sample_size=60, seed=42):
    by_id = {row['query_id']: row for row in queries}
    chosen = []
    per_type = max(1, sample_size // len(QUERY_TYPES))
    for kind in QUERY_TYPES:
        chosen.extend(sorted(row['query_id'] for row in queries if row['query_type'] == kind)[:per_type])
    if len(chosen) < sample_size:
        chosen.extend(key for key in sorted(by_id) if key not in chosen)[:sample_size - len(chosen)]
    chosen = set(chosen[:sample_size])
    labels = {'vector': 'System A', 'vector_reranker': 'System B',
              'vector_graph': 'System C', 'vector_graph_reranker': 'System D'}
    rows = []
    for item in rag_rows:
        if item['query_id'] not in chosen:
            continue
        query = by_id[item['query_id']]
        rows.append({'review_id': sha256(f'{item["query_id"]}:{item["system"]}'.encode()).hexdigest()[:16],
                     'query_id': item['query_id'], 'query': query['query'],
                     'query_type': query['query_type'], 'system_label': labels[item['system']],
                     'answer': item['answer'], 'expected_answer': query['expected_answer'],
                     'scores': {'correctness': None, 'relevance': None, 'faithfulness': None,
                                'citation_usefulness': None, 'temporal_correctness': None},
                     'reviewer': '', 'reviewed_at': '', 'notes': ''})
    random.Random(seed).shuffle(rows)
    return rows, labels


def markdown(report):
    lines = ['# CogniMem graph and reranker ablation', '',
             '**Research question:** ' + report['research_question'], '',
             f"Dataset: {report['dataset']['queries']} developer-labeled queries across {len(QUERY_TYPES)} types. Results are reproducible development evidence, not independent human validation.", '',
             '| System | Recall@5 | Recall@10 | MRR | nDCG@10 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for system, values in report['retrieval']['summary'].items():
        lines.append(f"| {system.replace('_', ' + ')} | {values['recall_at_5']['mean']:.2%} | {values['recall_at_10']['mean']:.2%} | {values['mrr']['mean']:.3f} | {values['ndcg_at_10']['mean']:.3f} |")
    lines += ['', '| System | Correctness | Faithfulness | Context precision | Citation completeness |',
              '| --- | ---: | ---: | ---: | ---: |']
    for system, values in report['rag']['summary'].items():
        lines.append(f"| {system.replace('_', ' + ')} | {values['correctness']['mean']:.2%} | {values['faithfulness']['mean']:.2%} | {values['context_precision']['mean']:.2%} | {values['citation_completeness']['mean']:.2%} |")
    lines += ['', '## Paired comparisons', '']
    for name, comparison in report['comparisons'].items():
        value = comparison['recall_at_5']
        lines.append(f"- {name}: recall@5 absolute change {value['absolute_improvement']:.4f}; 95% CI {value['ci95']}; paired randomization p={value['paired_randomization_p']:.4f}.")
    lines += ['', '## Recall@5 by query type', '',
              '| Query type | Vector | Vector + reranker | Vector + graph | Full |',
              '| --- | ---: | ---: | ---: | ---: |']
    for kind, systems in report['retrieval']['by_query_type'].items():
        values = [systems[system]['recall_at_5'] for system in SYSTEMS]
        rendered = [('n/a' if value is None else f'{value:.2%}') for value in values]
        lines.append(f'| {kind} | ' + ' | '.join(rendered) + ' |')
    lines += ['', '## End-to-end latency', '', '| System | P50 ms | P95 ms | P99 ms |',
              '| --- | ---: | ---: | ---: |']
    for system, stages in report['latency_ms'].items():
        values = stages['end_to_end']
        lines.append(f"| {system.replace('_', ' + ')} | {values['p50']:.2f} | {values['p95']:.2f} | {values['p99']:.2f} |")
    lines += ['', 'Per-query predictions, query-type tables, stage latency percentiles, failure cases, and all other metrics are retained in the JSON report.', '',
              'The blind human-review packet contains blank ratings. Human evaluation remains pending until independent reviewers complete it.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--output-dir', type=Path, default=Path('docs/reports/ablation'))
    parser.add_argument('--generator', choices=['extractive', 'gemini'], default='extractive')
    parser.add_argument('--env-file', default='.env')
    parser.add_argument('--human-sample', type=int, default=60)
    args = parser.parse_args()
    payload = load_dataset(args.dataset)
    client = gemini_client(env_file=args.env_file) if args.generator == 'gemini' else None
    started = time.perf_counter()
    with TemporaryDirectory() as directory:
        embedder = FastEmbedder()
        rags = {}
        for system in SYSTEMS:
            store = LocalMemoryStore(Path(directory) / f'{system}.jsonl')
            for item in payload['memories']:
                store.ingest(MemoryRecord.from_dict(item))
            rags[system] = MemoryRAG(store, embedder=embedder)
            rags[system].sync(user_id='gold-user')
        retrieval_rows, rag_rows = [], []
        # Keep historical snapshots together so every isolated system pays the
        # same index transition cost instead of whichever system happens to run first.
        queries = sorted(payload['queries'], key=lambda row: (row['query_type'] == 'historical', row['query_id']))
        for index, row in enumerate(queries, 1):
            retrieval_rows.extend(evaluate_retrieval(rags, row))
            rag_rows.extend(evaluate_rag(rags, row, generator=args.generator, client=client))
            if index % 20 == 0:
                print(json.dumps({'completed_queries': index, 'total': len(payload['queries'])}), flush=True)
    retrieval_summary = summarize(retrieval_rows, RETRIEVAL_METRICS)
    rag_summary = summarize(rag_rows, RAG_METRICS)
    comparisons = {}
    for name, left, right in (
        ('graph_contribution', 'vector', 'vector_graph'),
        ('reranker_on_vector', 'vector', 'vector_reranker'),
        ('reranker_on_graph', 'vector_graph', 'vector_graph_reranker')):
        comparisons[name] = {metric: paired_comparison(retrieval_rows, left, right, metric)
                             for metric in RETRIEVAL_METRICS}
    failures = sorted((row for row in retrieval_rows if row.get('recall_at_5') is not None),
                      key=lambda row: (row['recall_at_5'], row['ndcg_at_5']))[:50]
    report = {'schema_version': 1, 'created_at': datetime.now().astimezone().isoformat(),
              'research_question': payload['research_question'],
              'controls': payload['controls'],
              'dataset': {'path': str(args.dataset), 'sha256': sha256(args.dataset.read_bytes()).hexdigest(),
                          'queries': len(payload['queries']), 'memories': len(payload['memories']),
                          'label_provenance': payload['label_provenance']},
              'systems': SYSTEMS, 'retrieval': {'summary': retrieval_summary,
                  'by_query_type': summarize_types(retrieval_rows, RETRIEVAL_METRICS), 'rows': retrieval_rows},
              'rag': {'generator': args.generator, 'summary': rag_summary,
                  'by_query_type': summarize_types(rag_rows, RAG_METRICS), 'rows': rag_rows},
              'comparisons': comparisons, 'latency_ms': latency_summary(retrieval_rows, rag_rows),
              'failure_cases': failures, 'elapsed_seconds': time.perf_counter() - started,
              'memory_quality': quality_audit(payload),
              'lifecycle': evaluate_lifecycle(),
              'limitations': ['Gold labels are developer-authored templates.',
                  'Extractive correctness is deterministic string coverage, not human answer judgment.',
                  'Independent human evaluation is pending.',
                  'The FastEmbed-supported Apache-2.0 Xenova MiniLM reranker is used; BAAI/bge-reranker-v2-m3 is not supported by the pinned FastEmbed release.']}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output_dir / 'results.md').write_text(markdown(report))
    write_recall_chart(retrieval_summary, args.output_dir / 'recall_at_5.svg')
    reviews, mapping = make_human_packet(rag_rows, payload['queries'], sample_size=args.human_sample)
    (args.output_dir / 'human_review_blind.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in reviews))
    (args.output_dir / 'human_review_mapping.json').write_text(json.dumps(mapping, indent=2) + '\n')
    print(markdown(report))


if __name__ == '__main__':
    main()
