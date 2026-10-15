"""Run every CogniMem measurement in order and snapshot the reports under a tag.

    python scripts/measure_all.py --tag baseline --candidate-limit 20 --ann-arms
    python scripts/measure_all.py --tag e2 --candidate-limit 100 --skip-retention
    python scripts/measure_all.py --tag hybrid --scorer-steps-only            # only what the importance scorer affects
    python scripts/measure_all.py --tag heuristic_control --scorer-steps-only --scorer heuristic

Each step runs as its own process.
A failed step is logged under docs/reports/logs/ and later steps still run. The snapshot
goes to docs/reports/<tag>/ with a MANIFEST.json, and summary.json is also copied to the
tracked evaluation/reports/<tag>_summary.json because docs/ is git-ignored.

Only reports written by this run reach the snapshot. Each step's previous output is removed
before it runs, outputs of steps outside the plan are removed, and the reports of steps
skipped with --skip-locomo/--skip-retention are kept and recorded as reused.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'docs' / 'reports'
PY = sys.executable
# Files and directories under docs/reports/ that each step writes.
OUTPUTS = {'ablation_test': ('ablation',), 'ablation_dev': ('ablation_dev',), 'ablation_test_ann': ('ablation_ann',),
           'locomo': ('locomo_retrieval.json',), 'locomo_ann': ('locomo_retrieval_ann.json',),
           'retention': ('longmemeval_retention.json',), 'classification': ('conversational_evaluation.json',),
           'lifecycle': ('lifecycle.json',), 'report': ('summary.json', 'SUMMARY.md'), 'build_dataset': ()}


def steps(args) -> list[tuple[str, list[str]]]:
    shared = ['--candidate-limit', str(args.candidate_limit), '--top-k', str(args.top_k)]
    if args.ann:
        shared.append('--ann')
    if args.abstain_threshold is not None:
        shared += ['--abstain-threshold', str(args.abstain_threshold)]
    if args.structure != 'default':
        shared += ['--structure', args.structure]
    if args.scorer_steps_only:
        # The ablation and LoCoMo runners never call the importance scorer (fixed importance per record),
        # so a scorer change is measured by retention, classification and lifecycle alone.
        plan = []
    else:
        plan = [('build_dataset', [PY, '-m', 'evaluation.build_dataset', '--check']),
                ('ablation_test', [PY, '-m', 'evaluation.ablation', '--split', 'test', '--output-dir', 'docs/reports/ablation', *shared]),
                ('ablation_dev', [PY, '-m', 'evaluation.ablation', '--split', 'dev', '--output-dir', 'docs/reports/ablation_dev', *shared])]
    if args.ann_arms and not args.scorer_steps_only:
        ann = [a for a in shared if a != '--ann'] + ['--ann']
        plan.append(('ablation_test_ann', [PY, '-m', 'evaluation.ablation', '--split', 'test',
                                           '--output-dir', 'docs/reports/ablation_ann', *ann]))
    if not args.skip_locomo and not args.scorer_steps_only:
        locomo = [PY, '-m', 'evaluation.locomo', '--candidate-limit', str(args.candidate_limit)]
        if args.ann:
            locomo.append('--ann')
        if args.structure != 'default':
            locomo += ['--structure', args.structure]
        if not (ROOT / 'data/evaluation/locomo/locomo10.json').exists() and not args.skip_download:
            locomo.append('--download')
        plan.append(('locomo', locomo))
        if args.ann_arms:
            exact = [a for a in locomo if a not in ('--ann', '--download')]
            plan.append(('locomo_ann', [*exact, '--ann', '--output', 'docs/reports/locomo_retrieval_ann.json']))
    if not args.skip_retention:
        retention = [PY, '-m', 'evaluation.retention']
        if not (ROOT / 'data/evaluation/longmemeval/oracle.json').exists() and not args.skip_download:
            retention.append('--download')
        plan.append(('retention', retention))
    plan += [('classification', [PY, '-m', 'evaluation.classification']),
             ('lifecycle', [PY, '-m', 'evaluation.lifecycle']),
             ('report', [PY, '-m', 'evaluation.report'])]
    return plan


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remove(name: str) -> None:
    path = REPORTS / name
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def prepare(plan, args) -> dict:
    """Remove reports this run does not produce, except the intentionally reused ones."""
    planned = {name for name, _ in plan}
    reused_steps = {'locomo'} if args.skip_locomo else set()
    if args.skip_retention:
        reused_steps.add('retention')
    reused = {}
    for step, names in OUTPUTS.items():
        if step in planned:
            continue
        for name in names:
            path = REPORTS / name
            if step in reused_steps and path.is_file():
                try:
                    created = json.loads(path.read_text()).get('created_at')
                except ValueError:
                    created = None
                reused[name] = {'step': step, 'created_at': created, 'sha256': sha256(path)}
            elif step not in reused_steps:
                remove(name)
    shutil.rmtree(REPORTS / 'logs', ignore_errors=True)
    return reused


def git_state() -> dict:
    def run(*command):
        return subprocess.run(['git', *command], cwd=ROOT, capture_output=True).stdout
    untracked = run('ls-files', '--others', '--exclude-standard').decode().split()
    return {'git_head': run('rev-parse', '--short', 'HEAD').decode().strip(),
            'dirty_tree': bool(run('status', '--porcelain').strip()),
            'diff_sha256': hashlib.sha256(run('diff', 'HEAD', '--binary')).hexdigest(),
            'untracked': {name: sha256(ROOT / name) for name in untracked
                          if (ROOT / name).is_file() and not name.startswith('docs/')}}


def snapshot(tag: str, outcomes: list[dict], reused: dict, scorer: str) -> Path:
    target = REPORTS / tag
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for item in REPORTS.iterdir():
        if item == target or (item.is_dir() and (item / 'MANIFEST.json').exists()):
            continue
        destination = target / item.name
        shutil.copytree(item, destination) if item.is_dir() else shutil.copy2(item, destination)
    dataset = ROOT / 'evaluation/datasets/gold_queries.json'
    manifest = {'tag': tag, 'created_at': datetime.now().astimezone().isoformat(), 'argv': sys.argv, **git_state(),
                'dataset_sha256': sha256(dataset) if dataset.exists() else None, 'scorer': scorer, 'steps': outcomes, 'reused': reused,
                'files': {str(p.relative_to(target)): sha256(p) for p in sorted(target.rglob('*')) if p.is_file()}}
    (target / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    summary = REPORTS / 'summary.json'
    if summary.exists():
        tracked = ROOT / 'evaluation' / 'reports' / f'{tag}_summary.json'
        payload = json.loads(summary.read_text())
        payload['manifest'] = {k: manifest[k] for k in ('tag', 'created_at', 'argv', 'git_head', 'dirty_tree', 'diff_sha256',
                                                        'dataset_sha256', 'scorer', 'steps', 'reused')}
        tracked.write_text(json.dumps(payload, indent=2) + '\n')
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--candidate-limit', default='20')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--ann', action='store_true', help='run every step with approximate search')
    parser.add_argument('--ann-arms', action='store_true', help='also run separate approximate-search arms')
    parser.add_argument('--abstain-threshold', type=float, default=None)
    parser.add_argument('--structure', default='default',
                        help="post-rerank reordering in every step: 'default', 'none', or a step list")
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--skip-locomo', action='store_true')
    parser.add_argument('--skip-retention', action='store_true',
                        help='retention does not depend on retrieval settings; reuse the previous report '
                             '(it must come from the same importance scorer)')
    parser.add_argument('--scorer-steps-only', action='store_true',
                        help='run only the steps that use the importance scorer (retention, classification, lifecycle)')
    parser.add_argument('--scorer', choices=('hybrid', 'heuristic'), default=None,
                        help='importance scorer for every step (sets COGNIMEM_SCORER; default: the environment, else hybrid)')
    parser.add_argument('--overwrite', action='store_true', help='replace an existing snapshot with the same tag')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.dry_run and not args.overwrite and (
            (REPORTS / args.tag).exists() or (ROOT / 'evaluation' / 'reports' / f'{args.tag}_summary.json').exists()):
        parser.error(f'tag {args.tag!r} already exists; pass --overwrite to replace it')
    plan = steps(args)
    scorer = args.scorer or os.getenv('COGNIMEM_SCORER') or 'hybrid'
    env = {**os.environ, 'COGNIMEM_SCORER': scorer}
    logs = REPORTS / 'logs'
    reused = {}
    if not args.dry_run:
        reused = prepare(plan, args)
        for name, item in reused.items():
            if item['step'] == 'retention':
                stamp = json.loads((REPORTS / name).read_text()).get('importance_model')
                item['scorer'] = stamp.get('type') if isinstance(stamp, dict) else 'heuristic'   # older reports: heuristic
                if item['scorer'] != scorer:
                    parser.error(f'--skip-retention would reuse a {item["scorer"]} retention report in a {scorer} run; '
                                 f'rerun retention or pass --scorer {item["scorer"]}')
        logs.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for name, command in plan:
        print(f'[{name}] {" ".join(command[1:])}', flush=True)
        if args.dry_run:
            continue
        for output in OUTPUTS.get(name, ()):
            remove(output)   # a failed step must not leave the previous run's report behind
        started = time.perf_counter()
        with (logs / f'{name}.log').open('w') as handle:
            code = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env).returncode
        outcomes.append({'step': name, 'command': command[1:], 'exit_code': code,
                         'seconds': round(time.perf_counter() - started, 1)})
        print(f'[{name}] exit {code} in {outcomes[-1]["seconds"]} s', flush=True)
    if args.dry_run:
        return
    target = snapshot(args.tag, outcomes, reused, scorer)
    failed = [o['step'] for o in outcomes if o['exit_code']]
    print(f'Snapshot: {target.relative_to(ROOT)}' + (f'; FAILED steps: {failed}' if failed else ''))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
