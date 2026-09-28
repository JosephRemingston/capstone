"""Frozen external LoCoMo retrieval benchmark; never index QA or generated summaries.

Run: python -m evaluation.benchmark --download
This evaluates evidence retrieval, not answer correctness or semantic entailment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import time
import urllib.request

from main.retrieval.embeddings import FastEmbedder, normalized
from main.domain.models import MemoryRecord, MemoryCategory, MemoryTier
from main.storage.store import LocalMemoryStore
from main.retrieval.rag import MemoryRAG

COMMIT = '3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376'
SHA256 = '79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4'
ROOT = Path('data/evaluation/locomo')
MODES = ('recency', 'keyword', 'semantic', 'graph', 'hybrid')
PROTOCOL = {
    'dataset': 'LoCoMo', 'source': 'https://github.com/snap-research/locomo',
    'commit': COMMIT, 'sha256': SHA256, 'split': 'all 10 external conversations, no fitting',
    'k': 5, 'rrf_constant': 60, 'weights': {'keyword': .3, 'semantic': .5, 'graph': .2},
    'adversarial_category_5': 'Included in overall reference-evidence retrieval; reported separately from answerable categories. Evidence retrieval does not imply these questions are answerable.',
    'dense_min_cosine': .35, 'graph_hops': 2, 'context_characters': 10000,
    'excluded': 'questions without evidence, evidence IDs absent from text turns, or image-only evidence turns',
    'indexed': 'conversation text turns, speaker, session date only; no QA, observations, event/session summaries',
    'personalization_proxy': 'questions matching prefer|favorite|favourite|like|enjoy|hobby|hobbies (word boundaries)',
    'limits': 'Retrieval evidence metrics; not answer accuracy. No independent conflict-decision labels or human personalization ratings in this dataset. Developer graph/conflict tests are reported separately.',
}


def download(root=ROOT):
    root.mkdir(parents=True, exist_ok=True)
    base = f'https://raw.githubusercontent.com/snap-research/locomo/{COMMIT}'
    for remote, local in (('data/locomo10.json', 'locomo10.json'), ('LICENSE.txt', 'LICENSE.txt')):
        request = urllib.request.Request(f'{base}/{remote}', headers={'User-Agent': 'CogniMem-evaluation'})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        if local.endswith('.json') and hashlib.sha256(payload).hexdigest() != SHA256:
            raise ValueError('Dataset checksum mismatch')
        (root / local).write_bytes(payload)
    (root / 'manifest.json').write_text(json.dumps(PROTOCOL, indent=2) + '\n')


def conversation_records(row):
    result, image_ids = [], set()
    conversation = row['conversation']
    owner = str(row['sample_id'])
    for key in sorted(conversation):
        if not re.fullmatch(r'session_\d+', key):
            continue
        date = conversation[key + '_date_time']
        when = datetime.strptime(date, '%I:%M %p on %d %B, %Y').replace(tzinfo=timezone.utc)
        for turn in conversation[key]:
            if not isinstance(turn.get('text'), str) or not turn['text'].strip():
                image_ids.add(turn['dia_id'])
                continue
            result.append(MemoryRecord(id=turn['dia_id'], content=turn['text'], user_id=owner,
                session_id=key, category=MemoryCategory.SEMANTIC, tier=MemoryTier.LONG_TERM,
                created_at=when, updated_at=when, recorded_at=when,
                source_metadata={'speaker': turn['speaker'], 'session_date': date}))
    return result, image_ids


def metrics(predicted, gold, k=5):
    predicted = list(dict.fromkeys(predicted))[:k]
    gold = set(gold)
    if not gold:
        raise ValueError('Evidence labels must not be empty')
    relevant = [int(identifier in gold) for identifier in predicted]
    hits = sum(relevant)
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(relevant))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(gold))))
    return {'recall_at_5': hits / len(gold), 'hit_at_5': float(hits > 0),
            'mrr_at_5': next((1 / (rank + 1) for rank, hit in enumerate(relevant) if hit), 0),
            'ndcg_at_5': dcg / ideal}


class SnapshotStore(LocalMemoryStore):
    """Read-only benchmark snapshot; same retrieval semantics without repeated disk I/O."""
    def __init__(self, path, records):
        super().__init__(path)
        self.records = records
    def all(self):
        return self.records
    def _read_log(self):
        return self.records


class CachedEmbedder:
    def __init__(self, inner, queries):
        self.inner, self.model_id, self.dimension = inner, inner.model_id, inner.dimension
        vectors = inner.queries(queries)
        self.cache = {query: normalized(vector, self.dimension) for query, vector in zip(queries, vectors)}
    def passages(self, texts):
        return self.inner.passages(texts)
    def query(self, text):
        return self.cache[text]


def summarize(results):
    report = {}
    for mode in MODES:
        selected = [r for r in results if r['mode'] == mode]
        if not selected:
            continue
        measures = ['recall_at_5', 'hit_at_5', 'mrr_at_5', 'ndcg_at_5', 'context_characters']
        means = {key: statistics.mean(r[key] for r in selected) for key in measures}
        groups = defaultdict(list)
        for row in selected:
            groups[row['conversation']].append(row['recall_at_5'])
        group_values = list(groups.values())
        rng = random.Random(42)
        boot = []
        for _ in range(1000):
            sample = [value for group in rng.choices(group_values, k=len(group_values)) for value in group]
            boot.append(statistics.mean(sample))
        boot.sort()
        means['recall_95pct_conversation_bootstrap'] = [boot[25], boot[974]]
        pref = [r for r in selected if r['personalization_proxy']]
        means['personalization_proxy_count'] = len(pref)
        means['personalization_evidence_recall'] = statistics.mean(r['recall_at_5'] for r in pref) if pref else None
        means['by_category'] = {category: {'count': sum(r['category'] == category for r in selected),
            'recall_at_5': statistics.mean(r['recall_at_5'] for r in selected if r['category'] == category)}
            for category in sorted({r['category'] for r in selected})}
        report[mode] = means
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('docs/reports/locomo_retrieval.json'))
    args = parser.parse_args()
    if args.download:
        download()
    payload = (ROOT / 'locomo10.json').read_bytes()
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError('Dataset checksum mismatch')
    rows = json.loads(payload)
    embedder = FastEmbedder()
    results, efficiency, excluded = [], [], Counter()
    started = time.monotonic()
    for row in rows:
        records, image_ids = conversation_records(row)
        identifiers = {record.id for record in records}
        questions = []
        for index, qa in enumerate(row['qa']):
            gold = qa.get('evidence', [])
            if not gold:
                excluded['no_evidence'] += 1
            elif not all(key in identifiers for key in gold):
                excluded['missing_text_evidence'] += 1
            else:
                questions.append((index, qa))
        query_embedder = CachedEmbedder(embedder, list(dict.fromkeys(qa['question'] for _, qa in questions)))
        path = ROOT / 'indexes' / (str(row['sample_id']) + '.jsonl')
        store = SnapshotStore(path, records)
        rag = MemoryRAG(store, embedder=query_embedder)
        owner = str(row['sample_id'])
        rag.sync(user_id=owner)
        for index, qa in questions:
            for mode in MODES:
                hits = rag.search(qa['question'], user_id=owner, mode=mode, limit=5)
                predicted = [hit['memory_id'] for hit in hits]
                result = {'conversation': owner, 'question_index': index, 'mode': mode,
                          'category': str(qa['category']), 'predicted': predicted,
                          'personalization_proxy': bool(re.search(r'\b(prefer|favorite|favourite|like|enjoy|hobby|hobbies)\b', qa['question'], re.I)),
                          'context_characters': sum(len(hit['text']) for hit in hits),
                          **metrics(predicted, qa['evidence'])}
                results.append(result)
        efficiency.append({'conversation': owner, 'turns': len(records), 'retained_turns': len(store.all()),
                           'source_text_bytes': sum(len(r.content.encode()) for r in records),
                           'index_bytes': rag.database.path.stat().st_size,
                           'scored_questions': len(questions), 'image_only_turns_excluded': len(image_ids),
                           'consolidation_ratio': 1.0})
        print(json.dumps({'finished_conversation': owner, 'questions': len(questions),
                          'elapsed_seconds': round(time.monotonic() - started)}), flush=True)
    report = {'protocol': PROTOCOL, 'embedding_model': embedder.model_id, 'excluded_questions': dict(excluded),
              'scored_questions': len(results) // len(MODES), 'metrics': summarize(results),
              'answerable_metrics': summarize([r for r in results if r['category'] != '5']),
              'memory_efficiency': efficiency, 'elapsed_seconds': time.monotonic() - started,
              'independent_conflict_decisions': 'not measured: no gold conflict decisions in LoCoMo',
              'generated_answer_quality': 'not measured: Gemini credentials configured later by user',
              'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'results', 'memory_efficiency'}}, indent=2))


if __name__ == '__main__':
    main()

