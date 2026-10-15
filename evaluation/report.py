"""Render human-readable reports from the saved JSON results.

    python -m evaluation.report                  # docs/reports/SUMMARY.md and summary.json
    python -m evaluation.report --retention-only # only evaluation/reports/retention_summary.json

Every number comes from a saved JSON report; nothing is recomputed or typed by hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPORTS = Path('docs/reports')
TRAINING = Path('evaluation/reports')


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def pct(value) -> str:
    return '—' if value is None else f'{100 * value:.2f}%'


def num(value, digits=4) -> str:
    return '—' if value is None else f'{value:.{digits}f}'


def render_retention() -> dict:
    """Headline numbers of the default importance model from its training report (training/ensemble writes the .md)."""
    report = load(TRAINING / 'retention_ensemble.json')
    summary = {}
    if report:
        model = report['ensemble_thresholds']
        summary = {'ensemble_macro_f1': model['test']['macro_f1'], 'ensemble_accuracy': model['test']['accuracy'],
                   'ensemble_balanced_accuracy': model['test']['balanced_accuracy'],
                   'ensemble_sentence_ci95_low': model['sentence_ci95'][0], 'ensemble_sentence_ci95_high': model['sentence_ci95'][1],
                   'majority_macro_f1': report['majority']['test']['macro_f1'],
                   'oracle_macro_f1': report['oracle']['test']['macro_f1'],
                   'test_rows': sum(report['dataset']['class_counts']['test'].values()),
                   'test_sentences': report['dataset']['split_sentences']['test']}
    (TRAINING / 'retention_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            flatten(f'{prefix}.{key}' if prefix else str(key), item, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) or value is None:
        out[prefix] = value


def collect(reports: Path = REPORTS) -> dict:
    sections = {}
    for name in ('ablation', 'ablation_dev', 'ablation_ann'):
        report = load(reports / name / 'results.json')
        if report:
            sections[name] = {arm: {k: v for k, v in s.items() if k not in ('answerability', 'latency_ms')}
                              | {'answerable_accept_rate': s['answerability']['answerable_accept_rate'],
                                 'unanswerable_refusal_rate': s['answerability']['unanswerable_refusal_rate'],
                                 'balanced_answerability': s['answerability']['balanced_accuracy'],
                                 'p50_ms': s['latency_ms']['p50'], 'p95_ms': s['latency_ms']['p95']}
                              for arm, s in report['summary'].items()}
            sections[name]['_vector_paths'] = {arm: s.get('vector_paths') for arm, s in report['summary'].items()}
            sections[name]['_protocol'] = {'resolved_pool': report['protocol']['resolved_pool'],
                                           'ann': report['protocol']['ann'], 'top_k': report['protocol']['top_k'],
                                           'abstain_threshold': report['protocol']['abstain_threshold']}
    for name in ('locomo_retrieval', 'locomo_retrieval_ann'):
        report = load(reports / f'{name}.json')
        if report:
            sections[name] = {mode: {k: v for k, v in m.items() if k not in ('by_category', 'latency_ms',
                                                                                'recall_at_5_ci95_conversation_bootstrap')}
                              | {'p50_ms': m['latency_ms']['p50'], 'p95_ms': m['latency_ms']['p95'],
                                 **{f'{cat}_recall_at_5': v['recall_at_5'] for cat, v in m['by_category'].items()}}
                              for mode, m in report['metrics'].items()}
            if report.get('held_out_metrics'):
                # Conversations not used to choose the reranker pool size.
                sections[f'{name}_held_out'] = {mode: {k: v for k, v in m.items() if k in (
                    'questions', 'recall_at_5', 'recall_at_10', 'hit_at_5', 'mrr_at_5', 'pool_hit')}
                    for mode, m in report['held_out_metrics'].items() if mode.startswith('hybrid')}
            if report.get('vector_paths'):
                sections[name]['_vector_paths'] = report['vector_paths']
    retention = load(reports / 'longmemeval_retention.json')
    if retention:
        sections['longmemeval'] = {kind: {k: v for k, v in item.items()} for kind, item in retention['summary'].items()}
        sections['longmemeval']['_cleanup'] = retention['cleanup']
        if retention.get('diagnostics'):
            sections['longmemeval']['_diagnostics'] = retention['diagnostics']
    sections['_scorers'] = {name: report.get('importance_model') for name, report in (
        ('longmemeval', retention), ('classification', load(reports / 'conversational_evaluation.json')),
        ('lifecycle', load(reports / 'lifecycle.json'))) if report}
    classification = load(reports / 'conversational_evaluation.json')
    if classification:
        sections['classification'] = {k: classification[k] for k in ('cases', 'category_accuracy', 'category_macro_f1',
                                                                      'tier_accuracy', 'importance_range_agreement')}
        if classification.get('importance_range_baselines'):
            sections['classification']['_range_baselines'] = classification['importance_range_baselines']
    lifecycle = load(reports / 'lifecycle.json')
    if lifecycle:
        sections['lifecycle'] = {'checks': lifecycle['checks'], 'passed': lifecycle['passed']}
    # Numeric protocol markers, so report_delta warns when two snapshots used different scorers or clock handling.
    # Reports written before 16 Oct 2026 store importance_model as a string (or not at all): heuristic, clock unpinned.
    for name, report in (('longmemeval', retention), ('classification', classification), ('lifecycle', lifecycle)):
        if report and name in sections:
            stamp = report.get('importance_model')
            protocol = {'hybrid_scorer': int(isinstance(stamp, dict) and stamp.get('type') == 'hybrid')}
            if name != 'longmemeval':   # LongMemEval has always pinned its clock
                protocol['clock_pinned'] = int(isinstance(stamp, dict))
            sections[name]['_protocol'] = protocol
    model = load(TRAINING / 'retention_summary.json')
    if model:
        sections['importance_model'] = model
    flat = {}
    flatten('', sections, flat)
    return {'sections': sections, 'flat': flat}


def render_summary(data: dict) -> str:
    s = data['sections']
    lines = ['# CogniMem evaluation summary', '',
             'Generated by `python -m evaluation.report` from the JSON reports in this folder.', '']
    if 'ablation' in s:
        protocol = s['ablation']['_protocol']
        lines += [f"## Controlled benchmark, test split (pool {protocol['resolved_pool']}, exact={not protocol['ann']}, "
                  f"abstain τ={protocol['abstain_threshold']})", '',
                  '| Arm | Recall@5 | Recall@10 | MRR | Pool hit | Grounded correct | Unanswerable refused | Balanced | P50 ms |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for arm, m in s['ablation'].items():
            if arm.startswith('_'):
                continue
            lines.append(f"| {arm} | {pct(m['recall_at_5'])} | {pct(m['recall_at_10'])} | {num(m['mrr'], 3)} | "
                         f"{pct(m['pool_hit'])} | {pct(m['correct_grounded'])} | {pct(m['unanswerable_refusal_rate'])} | "
                         f"{pct(m['balanced_answerability'])} | {num(m['p50_ms'], 0)} |")
        lines.append('')
    for name, title in (('locomo_retrieval', 'LoCoMo evidence retrieval (exact search)'),
                        ('locomo_retrieval_ann', 'LoCoMo evidence retrieval (approximate search)')):
        if name in s:
            lines += [f'## {title}', '', '| Mode | Recall@5 | Recall@10 | Hit@5 | MRR@5 | Pool hit | Multi-hop R@5 | P50 ms |',
                      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
            for mode, m in s[name].items():
                if mode.startswith('_'):
                    continue
                lines.append(f"| {mode} | {pct(m['recall_at_5'])} | {pct(m['recall_at_10'])} | {pct(m['hit_at_5'])} | "
                             f"{num(m['mrr_at_5'], 3)} | {pct(m['pool_hit'])} | {pct(m.get('multi-hop_recall_at_5'))} | "
                             f"{num(m['p50_ms'], 0)} |")
            lines.append('')
    if 'locomo_retrieval_held_out' in s:
        lines += ['## LoCoMo held-out conversations (not used to choose the pool size)', '',
                  '| Mode | Questions | Recall@5 | Hit@5 | Pool hit |', '| --- | ---: | ---: | ---: | ---: |']
        for mode, m in s['locomo_retrieval_held_out'].items():
            lines.append(f"| {mode} | {m.get('questions', '—')} | {pct(m['recall_at_5'])} | {pct(m['hit_at_5'])} | "
                         f"{pct(m['pool_hit'])} |")
        lines.append('')
    if 'longmemeval' in s:
        lines += ['## LongMemEval evidence retention', '', '| Question type | Examples | Keep-everything | CogniMem |',
                  '| --- | ---: | ---: | ---: |']
        for kind, m in s['longmemeval'].items():
            if not kind.startswith('_'):
                lines.append(f"| {kind} | {m['examples']} | {pct(m['keep_everything_retention'])} | {pct(m['memory_retention'])} |")
        lines.append('')
    if 'classification' in s:
        c = s['classification']
        lines += ['## Classification cases', '', f"{c['cases']} Claude-authored cases: category accuracy {pct(c['category_accuracy'])}, "
                  f"macro F1 {num(c['category_macro_f1'], 3)}, tier accuracy {pct(c['tier_accuracy'])}, "
                  f"importance-range agreement {pct(c['importance_range_agreement'])}.", '']
    if 'lifecycle' in s:
        lines += [f"Lifecycle checks: {s['lifecycle']['passed']}/{s['lifecycle']['checks']} passed.", '']
    if s.get('_scorers'):
        describe = lambda stamp: stamp.get('type', 'unrecorded') if isinstance(stamp, dict) else (stamp or 'unrecorded')
        lines += ['Importance scorer per run: ' + '; '.join(
            f"{name} {describe(stamp)}" for name, stamp in s['_scorers'].items()) + '.', '']
    if 'importance_model' in s:
        m = s['importance_model']
        lines += ['## Importance model (csv5k test split)', '', '| Model | Macro F1 | Sentence CI | Accuracy | Balanced accuracy |',
                  '| --- | ---: | --- | ---: | ---: |',
                  f"| Ensemble + thresholds | {num(m.get('ensemble_macro_f1'))} | [{num(m.get('ensemble_sentence_ci95_low'), 3)}, "
                  f"{num(m.get('ensemble_sentence_ci95_high'), 3)}] | {pct(m.get('ensemble_accuracy'))} | {pct(m.get('ensemble_balanced_accuracy'))} |",
                  f"| Always the most common class | {num(m.get('majority_macro_f1'))} | — | — | — |",
                  f"| Best possible (oracle) | {num(m.get('oracle_macro_f1'))} | — | — | — |", '']
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--retention-only', action='store_true')
    parser.add_argument('--reports', type=Path, default=REPORTS)
    args = parser.parse_args()
    summary = render_retention()
    if args.retention_only:
        print(json.dumps(summary, indent=2))
        return
    data = collect(args.reports)
    markdown = render_summary(data)   # render first: a failure must not leave a half-written snapshot behind
    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / 'summary.json').write_text(json.dumps(data, indent=2) + '\n')
    (args.reports / 'SUMMARY.md').write_text(markdown)
    print(markdown)


if __name__ == '__main__':
    main()
