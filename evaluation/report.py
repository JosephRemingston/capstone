"""Render human-readable reports from the saved JSON results.

    python -m evaluation.report                  # docs/reports/SUMMARY.md and summary.json
    python -m evaluation.report --retention-only # evaluation/reports/*.md from the training JSON

Every number comes from a saved JSON report; nothing is recomputed or typed by hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPORTS = Path('docs/reports')
TRAINING = Path('evaluation/reports')
CLASSES = ('invalid', 'short_term', 'long_term')


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def pct(value) -> str:
    return '—' if value is None else f'{100 * value:.2f}%'


def num(value, digits=4) -> str:
    return '—' if value is None else f'{value:.{digits}f}'


def class_tables(metrics: dict) -> list[str]:
    lines = ['| Gold class | Support | Precision | Recall | F1 |', '| --- | ---: | ---: | ---: | ---: |']
    for name in CLASSES:
        item = metrics['per_class'][name]
        lines.append(f"| {name} | {item['support']} | {num(item['precision'])} | {num(item['recall'])} | {num(item['f1'])} |")
    lines += ['', '| Gold \\ predicted | ' + ' | '.join(CLASSES) + ' |', '| --- | ---: | ---: | ---: |']
    for name, row in zip(CLASSES, metrics['confusion_matrix']):
        lines.append(f'| {name} | ' + ' | '.join(str(v) for v in row) + ' |')
    return lines


def render_xgboost(report: dict) -> str:
    test, base = report['test'], report['majority_baseline']
    lines = ['# XGBoost conversational retention model', '',
             f"Dataset: {report['dataset']}. Target: {report['target']}. Model selection: {report['selection']}.", '',
             f"Rows used: {report['dataset_size']['train_used']} train, {report['dataset_size']['test_used']} test "
             f"({report['dataset_size']['train_omitted']} train row omitted).", '',
             '## Validation trials', '', '| max_depth | min_child_weight | Best iteration | Macro F1 | Balanced accuracy |',
             '| ---: | ---: | ---: | ---: | ---: |']
    for trial in report['validation_trials']:
        lines.append(f"| {trial['max_depth']} | {trial['min_child_weight']} | {trial['best_iteration']} | "
                     f"{num(trial['macro_f1'])} | {num(trial['balanced_accuracy'])} |")
    hyper = report['selected_hyperparameters']
    lines += ['', f"Selected: max_depth {hyper['max_depth']}, min_child_weight {hyper['min_child_weight']}, "
              f"{hyper['n_estimators']} trees on refit.", '', '## Published test split', '',
              '| Metric | XGBoost | Always long-term |', '| --- | ---: | ---: |']
    for key in ('accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1'):
        lines.append(f'| {key} | {num(test[key])} | {num(base.get(key))} |')
    lines.append(f"| proxy_mae | {num(test['proxy_mae'])} | — |")
    lines += [''] + class_tables(test) + ['', 'Reproduce: `python -m training.train_conversational_retention`.']
    return '\n'.join(lines) + '\n'


def render_neural(report: dict, depth: dict | None, *, artifact: str) -> str:
    test, xgb = report['neural_test'], report['xgboost_test']
    lines = [f"# Neural conversational retention model ({report['architecture']})", '',
             f"Dataset: {report['dataset']}. Fit/validation/test: {report['fit_size']}/{report['validation_size']}/"
             f"{report['test_size']}. Artifact: `{artifact}`.", '',
             '## Seeds (validation)', '', '| Seed | Best epoch | Macro F1 | NLL |', '| ---: | ---: | ---: | ---: |']
    for seed in report['seeds']:
        lines.append(f"| {seed['seed']} | {seed['best_epoch']} | {num(seed['validation_macro_f1'])} | {num(seed['validation_nll'])} |")
    lines += ['', f"Selected seed {report['selected_seed']}; temperature {num(report['temperature'])} reduced validation "
              f"NLL from {num(report['validation_nll_before_calibration'])} to {num(report['validation_nll_after_calibration'])}.",
              '', '## Published test split', '', '| Metric | Neural | XGBoost |', '| --- | ---: | ---: |']
    for key in ('accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'proxy_mae'):
        lines.append(f'| {key} | {num(test[key])} | {num(xgb[key])} |')
    lines += [f"| nll | {num(test['nll'])} | — |", f"| ece_10_bins | {num(test['ece_10_bins'])} | — |",
              f"| warm latency p50 / p95 ms | {num(report['latency_ms']['p50'], 2)} / {num(report['latency_ms']['p95'], 2)} | — |", '']
    lines += class_tables(test)
    if depth:
        lines += ['', '## Depth comparison (train split only)', '', '| Architecture | Parameters | Mean validation macro F1 |',
                  '| --- | ---: | ---: |']
        for name, item in depth['results'].items():
            lines.append(f"| {name} | {item['parameter_count']} | {num(item['mean_validation_macro_f1'])} |")
    lines += ['', f"Caveat: {report.get('caveat', '')}"]
    return '\n'.join(lines) + '\n'


def render_retention() -> dict:
    xgb = load(TRAINING / 'conversational_retention.json')
    deep = load(TRAINING / 'neural_retention.json')
    shallow = load(TRAINING / 'neural_retention_shallow.json')
    depth = load(TRAINING / 'neural_depth_validation.json')
    summary = {}
    if xgb:
        (TRAINING / 'conversational_retention.md').write_text(render_xgboost(xgb))
        summary.update(xgboost_macro_f1=xgb['test']['macro_f1'], xgboost_accuracy=xgb['test']['accuracy'],
                       majority_macro_f1=xgb['majority_baseline']['macro_f1'],
                       xgboost_invalid_recall=xgb['test']['per_class']['invalid']['recall'])
    for name, report, artifact in (('deep', deep, 'artifacts/importance_neural'),
                                   ('shallow', shallow, 'artifacts/importance_neural_shallow')):
        if report:
            out = 'neural_retention.md' if name == 'deep' else 'neural_retention_shallow.md'
            (TRAINING / out).write_text(render_neural(report, depth, artifact=artifact))
            test = report['neural_test']
            summary.update({f'{name}_macro_f1': test['macro_f1'], f'{name}_accuracy': test['accuracy'],
                            f'{name}_ece': test['ece_10_bins'], f'{name}_invalid_recall': test['per_class']['invalid']['recall']})
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
    classification = load(reports / 'conversational_evaluation.json')
    if classification:
        sections['classification'] = {k: classification[k] for k in ('cases', 'category_accuracy', 'category_macro_f1',
                                                                      'tier_accuracy', 'importance_range_agreement')}
    lifecycle = load(reports / 'lifecycle.json')
    if lifecycle:
        sections['lifecycle'] = {'checks': lifecycle['checks'], 'passed': lifecycle['passed']}
    models = load(TRAINING / 'retention_summary.json')
    if models:
        sections['retention_models'] = models
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
    if 'retention_models' in s:
        m = s['retention_models']
        lines += ['## Retention models (published test split)', '', '| Model | Macro F1 | Accuracy | Invalid recall |',
                  '| --- | ---: | ---: | ---: |',
                  f"| XGBoost | {num(m.get('xgboost_macro_f1'))} | {num(m.get('xgboost_accuracy'))} | {num(m.get('xgboost_invalid_recall'))} |",
                  f"| Neural (deep) | {num(m.get('deep_macro_f1'))} | {num(m.get('deep_accuracy'))} | {num(m.get('deep_invalid_recall'))} |",
                  f"| Neural (shallow) | {num(m.get('shallow_macro_f1'))} | {num(m.get('shallow_accuracy'))} | {num(m.get('shallow_invalid_recall'))} |",
                  f"| Always long-term | {num(m.get('majority_macro_f1'))} | — | 0 |", '']
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
    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / 'summary.json').write_text(json.dumps(data, indent=2) + '\n')
    (args.reports / 'SUMMARY.md').write_text(render_summary(data))
    print(render_summary(data))


if __name__ == '__main__':
    main()
