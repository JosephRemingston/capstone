"""Frozen external preference/update cases, baseline answers, and blind review.

Gold answers and evidence annotations are never passed to retrieval/generation.
Default answers are extractive; --generator gemini explicitly enables API calls.
Human scoring is a separate command; blank or stale reviews are never scores.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random
import statistics
from tempfile import TemporaryDirectory

from main.retrieval.embeddings import FastEmbedder
from main.retrieval.generation import answer, gemini_client
from main.storage.indexes import digest
from main.domain.models import MemoryRecord, MemoryCategory, MemoryTier
from main.retrieval.rag import MemoryRAG
from .benchmark import SnapshotStore
from .longmemeval import ROOT, SHA256, parse_date, chronology_valid

MODES = ('no_memory', 'keyword', 'hybrid')
RUBRIC = ('preference_alignment', 'groundedness', 'usefulness', 'conflict_correctness')


def records_for(row):
    if 'question_date' in row and not chronology_valid(row):
        raise ValueError('Future sessions must not enter answer context')
    records, gold = [], set()
    for date, session, turns in zip(row['haystack_dates'], row['haystack_session_ids'], row['haystack_sessions']):
        for index, turn in enumerate(turns):
            identifier = f'{session}:{index}'
            if turn.get('has_answer'):
                gold.add(identifier)
            if turn['content'].strip():
                when = parse_date(date)
                records.append(MemoryRecord(turn['content'], row['question_id'], session, MemoryCategory.SEMANTIC,
                    id=identifier, tier=MemoryTier.LONG_TERM, role=turn['role'], created_at=when,
                    updated_at=when, recorded_at=when,
                    source_metadata={'speaker': turn['role'], 'session_date': date}))
    return records, gold


def run(rows, *, generator='extractive', embedder=None):
    results, reviews, mapping = [], [], {}
    for row in rows:
        records, gold = records_for(row)
        with TemporaryDirectory() as directory:
            rag = MemoryRAG(SnapshotStore(Path(directory) / 'snapshot.jsonl', records), embedder=embedder)
            for mode in MODES:
                if mode == 'no_memory':
                    context = {'query': row['question'], 'sources': {}, 'paths': [], 'text': '',
                               'historical': False, 'characters': 0, 'retrieved': 0}
                else:
                    context = rag.context(row['question'], user_id=row['question_id'], mode=mode, limit=5)
                context['as_of'] = parse_date(row['question_date']).isoformat()
                context['known_at'] = context['as_of']
                try:
                    response = answer(context, generator=generator)
                    status = 'answered'
                except ValueError:
                    response, status = None, 'generation_failed'
                selected = set(context['sources'])
                answer_id = digest([row['question_id'], mode, generator])[:24]
                binding = digest({'id': answer_id, 'question': row['question'], 'reference': row['answer'],
                                  'response': response, 'sources': context['sources']})
                results.append({'question_id': row['question_id'], 'type': row['question_type'], 'mode': mode,
                    'answer_id': answer_id, 'binding': binding, 'status': status, 'response': response,
                    'evidence_recall': len(gold & selected) / len(gold) if gold else None,
                    'context_characters': context['characters'], 'source_count': len(selected)})
                mapping[answer_id] = {'question_id': row['question_id'], 'mode': mode}
                reviews.append({'answer_id': answer_id, 'binding': binding, 'question': row['question'],
                    'type': row['question_type'], 'reference_answer': row['answer'],
                    'response': response, 'retrieved_evidence': context['sources'],
                    'gold_evidence': {r.id: r.content for r in records if r.id in gold},
                    'reviewer': '', 'reviewed_at': '', 'independent': False,
                    'scores': {key: None for key in RUBRIC}, 'notes': ''})
    random.Random(42).shuffle(reviews)  # Blind mode names; keep mapping separate.
    summary = {}
    for mode in MODES:
        selected = [r for r in results if r['mode'] == mode]
        labeled = [r for r in selected if r['evidence_recall'] is not None]
        summary[mode] = {'examples': len(selected), 'labeled_examples': len(labeled),
            'evidence_recall': statistics.mean(r['evidence_recall'] for r in labeled) if labeled else None,
            'generation_failures': sum(r['status'] != 'answered' for r in selected),
            'mean_context_characters': statistics.mean(r['context_characters'] for r in selected) if selected else None}
    report = {'schema_version': 1, 'dataset_sha256': SHA256, 'generator': generator,
              'protocol': 'All oracle preference/update examples; no fitting; raw conversation documents only; k=5; 10000-character context; no-memory/keyword/hybrid baselines.',
              'limits': 'Oracle sessions omit distractors. Evidence recall is not personalization quality. Human review pending until a complete independent rating file is supplied. Extractive responses are snippets, not Gemini recommendations.',
              'human_quality_status': 'pending_independent_review', 'summary': summary, 'results': results}
    return report, reviews, mapping


def score_reviews(report, reviews):
    expected = {r['answer_id']: r for r in report['results'] if r['status'] == 'answered'}
    if not expected:
        raise ValueError('No successful answers to review')
    ratings, seen = defaultdict(list), set()
    for review in reviews:
        identifier = review.get('answer_id')
        reviewer = review.get('reviewer')
        if identifier not in expected or not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError('Review needs a known answer ID and reviewer identity')
        if (identifier, reviewer) in seen:
            raise ValueError('Duplicate reviewer/answer pair')
        seen.add((identifier, reviewer))
        actual_binding = digest({'id': identifier, 'question': review.get('question'),
                                 'reference': review.get('reference_answer'), 'response': review.get('response'),
                                 'sources': review.get('retrieved_evidence')})
        if (review.get('binding') != expected[identifier]['binding'] or actual_binding != review.get('binding')
                or review.get('independent') is not True):
            raise ValueError('Reviews must attest independence and match the frozen answer binding')
        if expected[identifier]['status'] != 'answered':
            raise ValueError('Failed generations cannot be scored as answers')
        try:
            when = datetime.fromisoformat(review['reviewed_at'].replace('Z', '+00:00'))
            if when.tzinfo is None or when > datetime.now(timezone.utc):
                raise ValueError('Invalid review time')
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError('Review requires a valid, nonfuture timezone-aware date') from exc
        kind = expected[identifier]['type']
        applicable = {'groundedness', 'usefulness',
                      'preference_alignment' if kind == 'single-session-preference' else 'conflict_correctness'}
        scores = review.get('scores', {})
        if not isinstance(scores, dict) or set(scores) != set(RUBRIC):
            raise ValueError('Review must use the complete rubric')
        for key, value in scores.items():
            if key in applicable and (type(value) is not int or value not in (0, 1, 2)):
                raise ValueError('Applicable scores must be integers 0, 1, or 2')
            if key not in applicable and value is not None:
                raise ValueError('Nonapplicable scores must be null')
        ratings[identifier].append(scores)
    if set(ratings) != set(expected):
        raise ValueError('Complete review coverage is required; unreviewed answers are not zero scores')
    scores_by_mode = defaultdict(lambda: defaultdict(list))
    disagreements = compared = 0
    for identifier, items in ratings.items():
        for key in RUBRIC:
            values = [r[key] for r in items if r[key] is not None]
            if not values:
                continue
            scores_by_mode[expected[identifier]['mode']][key].append(statistics.mean(values))
            if len(values) > 1:
                compared += 1
                disagreements += len(set(values)) > 1
    return {'status': 'independently_reviewed', 'answers': len(expected), 'ratings': len(reviews),
            'failed_answers_excluded': len(report['results']) - len(expected),
            'reviewer_attestation': 'Identity/independence are supplied attestations, not externally verified.',
            'mean_scores_0_to_2': {mode: {k: statistics.mean(v) for k,v in measures.items()} for mode,measures in scores_by_mode.items()},
            'multi_reviewer_dimensions': compared,
            'disagreement_fraction': disagreements / compared if compared else None}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--generator', choices=['extractive', 'gemini'], default='extractive')
    parser.add_argument('--output-dir', type=Path, default=Path('data/evaluation/personalization'))
    parser.add_argument('--reviews', type=Path, help='Score complete independent reviews of the existing report.')
    args = parser.parse_args()
    directory = args.output_dir
    if args.reviews:
        report = json.loads((directory / 'answers.json').read_text())
        reviews = [json.loads(line) for line in args.reviews.read_text().splitlines() if line.strip()]
        scored = score_reviews(report, reviews)
        scored['answers_report_sha256'] = sha256((directory / 'answers.json').read_bytes()).hexdigest()
        scored['review_file_sha256'] = sha256(args.reviews.read_bytes()).hexdigest()
        (directory / 'human_scores.json').write_text(json.dumps(scored, indent=2) + '\n')
        print(json.dumps(scored, indent=2))
        return
    if args.generator == 'gemini':
        gemini_client()  # Fail before benchmarking if credentials/dependencies are missing.
    payload = (ROOT / 'oracle.json').read_bytes()
    if sha256(payload).hexdigest() != SHA256:
        raise ValueError('Dataset checksum mismatch')
    requested = [r for r in json.loads(payload) if r['question_type'] in {'knowledge-update', 'single-session-preference'}]
    rows = [r for r in requested if chronology_valid(r)]
    report, reviews, mapping = run(rows, generator=args.generator, embedder=FastEmbedder())
    report['chronology_exclusions'] = [r['question_id'] for r in requested if not chronology_valid(r)]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'answers.json').write_text(json.dumps(report, indent=2) + '\n')
    (directory / 'blind_review.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in reviews))
    (directory / 'review_mapping.json').write_text(json.dumps(mapping, indent=2) + '\n')
    Path('docs/reports').mkdir(parents=True, exist_ok=True)
    Path('docs/reports/personalization.json').write_text(json.dumps({k:v for k,v in report.items() if k != 'results'}, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()

