"""External bAbI QA1 state-update check with a declared deterministic grammar adapter.

This isolates current-fact resolution and graph consistency. It does not measure
free-form conversational extraction. Gold answers are from the published test set.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import urllib.request

from main.application.core import MemoryCore
from main.domain.models import MemoryInput
from main.retrieval.rag import MemoryRAG
from main.storage.store import LocalMemoryStore

ROOT = Path('data/evaluation/babi')
REVISION = '821006e33d6ff3674fd888792a4dd1c04648f69f'
SHA256 = '9875dfad271cbfa5c49adb5809dd67be3826394a5d4d66dc74cab0c81483e2f8'
URL = f'https://huggingface.co/datasets/facebook/babi_qa/resolve/{REVISION}/en-qa1/test/0000.parquet'


def adapt(text):
    match = re.fullmatch(r'(\w+) (?:went|moved|journeyed|travelled) (?:back )?to the (\w+)\.', text)
    if not match:
        raise ValueError('Unsupported bAbI movement grammar; do not silently drop examples')
    return match[1], match[2], f'{match[1]} lives in {match[2]}.'


def evaluate(rows):
    counts, results = Counter(), []
    for story_index, row in enumerate(rows):
        story = row['story']
        first, latest = {}, {}
        with TemporaryDirectory() as directory:
            store = LocalMemoryStore(Path(directory) / 'log.jsonl')
            rag = MemoryRAG(store)
            for position, (identifier, text, gold) in enumerate(zip(story['id'], story['text'], story['answer'])):
                when = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=position)
                if '?' not in text:
                    name, location, adapted = adapt(text)
                    record = MemoryCore().process(MemoryInput(adapted, 'babi', str(story_index), timestamp=when))
                    record.id = str(identifier)
                    store.ingest(record)
                    first.setdefault(name, location)
                    if name in latest and latest[name] != location:
                        counts['changed_location_events'] += 1
                    latest[name] = location
                    continue
                match = re.fullmatch(r'Where is (\w+)\?', text)
                if not match:
                    raise ValueError('Unsupported bAbI question grammar')
                name = match[1]
                candidates = [r for r in store.list(user_id='babi', now=when)
                              if r.claim and r.claim['subject'] == 'entity:' + name.casefold()]
                current = [r.claim['value'] for r in candidates]
                rag.sync(user_id='babi', semantic=False)
                entities = rag.graph.entities(name, user_id='babi', exact=True)
                edges = [e for entity in entities for e in rag.graph.relations(user_id='babi', as_of=when,
                         entity_id=entity['id'], predicate='lives_in') if e['subject'] == entity['id']]
                graph = list({e['object_label'] for e in edges})
                result = {'story': story_index, 'question_id': str(identifier),
                          'memory_correct': current == [gold], 'graph_correct': graph == [gold],
                          'first_assertion_correct': first.get(name) == gold,
                          'latest_assertion_correct': latest.get(name) == gold}
                results.append(result)
            counts['input_events'] += sum('?' not in text for text in story['text'])
            counts['retained_roots'] += len(store.list(user_id='babi', now=when))
            counts['revisions'] += len(store._read_log())
    return {'stories': len(rows), 'questions': len(results), 'counts': dict(counts),
            'metrics': {key: sum(r[key] for r in results) / len(results) for key in (
                'memory_correct', 'graph_correct', 'first_assertion_correct', 'latest_assertion_correct')},
            'results': results}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('docs/reports/babi_conflicts.json'))
    args = parser.parse_args()
    if args.download:
        ROOT.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != SHA256:
            raise ValueError('Dataset checksum mismatch')
        (ROOT / 'test.parquet').write_bytes(payload)
    payload = (ROOT / 'test.parquet').read_bytes()
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError('Dataset checksum mismatch')
    import pyarrow.parquet as pq
    rows = pq.read_table(ROOT / 'test.parquet').to_pylist()
    report = {'dataset': 'https://huggingface.co/datasets/facebook/babi_qa', 'revision': REVISION,
              'sha256': SHA256, 'license': 'CC-BY-3.0',
              'protocol': 'All English QA1 test stories/questions; no fitting. Movement events mapped to supported named-person residence claims solely to isolate conflict resolution. Gold answers unchanged.',
              'limits': 'External synthetic state tracking with a grammar adapter, not human conversation or arbitrary language parsing. Latest-assertion baseline solves this task by construction. Not a substitute for independent human policy judgments.',
              **evaluate(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
