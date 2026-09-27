"""Run external evaluation stages and produce a single auditable report.

No API calls by default. --generator gemini explicitly opts into hosted answers.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/reports'


def report(stages):
    sources = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
               for folder in ('main', 'tests/evaluation') for p in sorted((ROOT / folder).glob('*.py'))}
    dependencies = {}
    for package in ('fastembed', 'langchain-google-genai', 'pyarrow'):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = 'not installed'
    result = {'implementation_sha256': sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
              'source_files': sources, 'python': sys.version.split()[0], 'dependencies': dependencies,
              'generated_at': datetime.now(timezone.utc).isoformat(), 'stages': stages,
              'independence': 'Externally authored frozen datasets, no tuning in this run. Prior LoCoMo/bAbI and 108 LongMemEval cases were evaluated before; 392 additional LongMemEval examples extend coverage. This is not a fresh held-out test after any future tuning.',
              'human_personalization_and_conflict_policy': 'pending independent review; never inferred from evidence recall',
              'files': {}}
    for name in ('locomo_retrieval', 'longmemeval_retention', 'babi_conflicts', 'personalization'):
        path = OUT / (name + '.json')
        stage = next((s for s in stages if s['name'] == name), None)
        if not stage or stage['status'] != 'passed':
            continue  # Never report stale files from a failed stage as fresh results.
        data = json.loads(path.read_text())
        result['files'][name] = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path.read_bytes()).hexdigest()}
        if name == 'locomo_retrieval':
            result['retrieval'] = {'questions': data['scored_questions'], 'metrics': data['metrics']}
        elif name == 'longmemeval_retention':
            result['retention'] = data['summary']
            result['chronology_exclusions'] = data.get('chronology_exclusions', {})
            result['physical_cleanup'] = data['cleanup']
        elif name == 'babi_conflicts':
            result['conflict_state_tracking'] = {'questions': data['questions'], 'metrics': data['metrics'], 'limits': data['limits']}
        else:
            result['personalization_evidence'] = data['summary']
            result['personalization_generator'] = data['generator']
    complete = {s['name'] for s in stages} == {'locomo_retrieval', 'longmemeval_retention', 'babi_conflicts', 'personalization'}
    result['execution_status'] = 'passed' if complete and all(s['status'] == 'passed' for s in stages) else 'incomplete'
    return result


def markdown(data):
    lines = ['# Independent evaluation and cleanup', '',
             'Execution status: **' + data['execution_status'] + '**. This means the evaluation ran, not that model quality passed a target.', '',
             data['independence'], '', '| Stage | Status | Seconds |', '| --- | --- | ---: |']
    for stage in data['stages']:
        lines.append(f"| {stage['name']} | {stage['status']} | {stage['seconds']:.1f} |")
    if 'retrieval' in data:
        lines += ['', '| Retriever | Reference recall@5 | Hit@5 |', '| --- | ---: | ---: |']
        for mode, values in data['retrieval']['metrics'].items():
            lines.append(f"| {mode} | {values['recall_at_5']:.2%} | {values['hit_at_5']:.2%} |")
        lines += ['', 'LoCoMo reference retrieval includes adversarial cases; answerable-only and per-category metrics are in the detailed report. Retrieval recall is not answer correctness.']
    if 'chronology_exclusions' in data:
        lines += ['', f"Excluded {data['chronology_exclusions'].get('count', 0)} LongMemEval cases whose sessions occur after the question; these cases do not enter retention/cleanup scoring."]
    if 'physical_cleanup' in data:
        c = data['physical_cleanup']
        lines += ['', f"Cleanup on isolated external replay stores removed {c['removed_memories']:,} eligible memories. JSONL size changed from {c['bytes_before']:,} to {c['bytes_after']:,} bytes. Visible memory state was unchanged in {c['visible_state_preserved']}/{c['examples']} cases."]
    if 'retention' in data:
        lines += ['', '| Conversation type | Labeled cases | Retained gold evidence | Retain-all baseline |', '| --- | ---: | ---: | ---: |']
        for kind, values in data['retention'].items():
            recall = values['memory_gold_evidence_retention']
            lines.append(f"| {kind} | {values['with_evidence_labels']} | {recall:.2%} | 100% |" if recall is not None else f"| {kind} | 0 | not measured | not measured |")
    if 'conflict_state_tracking' in data:
        lines += ['', 'bAbI uses a deterministic movement-to-location grammar adapter to isolate state updates; it is not natural-language conflict understanding.', '', '| State tracker | Accuracy |', '| --- | ---: |']
        for mode, value in data['conflict_state_tracking']['metrics'].items():
            lines.append(f'| {mode} | {value:.2%} |')
    if 'personalization_evidence' in data:
        lines += ['', 'Personalization/update answers use **' + data['personalization_generator'] + '** generation. Human quality scores are pending, not zero or assumed correct.', '', '| Answer context baseline | Evidence recall | Generation failures |', '| --- | ---: | ---: |']
        for mode, values in data['personalization_evidence'].items():
            lines.append(f"| {mode} | {values['evidence_recall']:.2%} | {values['generation_failures']} |")
    lines += ['', 'Blind review packets, frozen answer bindings, and baseline mappings are written to `data/evaluation/personalization/`. Complete independent reviews can be scored with `python -m tests.evaluation.personalization --reviews FILE`. Review assertions are supplied attestations; the software cannot certify reviewer independence.', '',
              'No cleanup was applied to the real user store, and no background service was installed by this evaluation run. See [cleanup operations](../cleanup.md) and [evaluation instructions](../../tests/evaluation/README.md).']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--generator', choices=['extractive', 'gemini'], default='extractive')
    args = parser.parse_args()
    stages = []
    commands = [('locomo_retrieval', 'benchmark', []),
                ('longmemeval_retention', 'longmemeval', ['--all-types']),
                ('babi_conflicts', 'conflicts', []),
                ('personalization', 'personalization', ['--generator', args.generator])]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, module, flags in commands:
        command = [sys.executable, '-m', 'tests.evaluation.' + module, *flags]
        if args.download and module != 'personalization':
            command.append('--download')
        started = time.monotonic()
        print('Running ' + name, flush=True)
        log = OUT / (name + '.run.log')
        with log.open('w') as stream:
            completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        stages.append({'name': name, 'status': 'passed' if completed.returncode == 0 else 'failed',
                       'seconds': time.monotonic() - started, 'exit_code': completed.returncode})
        data = report(stages)
        (OUT / 'independent_evaluation.json').write_text(json.dumps(data, indent=2) + '\n')
        (OUT / 'independent_evaluation.md').write_text(markdown(data))
    print(json.dumps({'execution_status': data['execution_status'], 'stages': stages}, indent=2))
    return 0 if data['execution_status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
