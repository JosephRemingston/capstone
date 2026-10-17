"""Replay collected contexts through four abstention rules.

  cd repo && PYTHONPATH=. python ../tools/replay.py ../out/ctx_calib.json [--show RULE]
Rules: none (never abstain), f6 (frozen F6 generation.py), prior (refuse any F6 slot-pattern question),
v2 (this repo's generation.abstention_reason with its default threshold argument).
Context files from tools/collect.py (benchmark/calibration) or tools/external.py (LongMemEval/LoCoMo).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from main.retrieval import generation as v2

F6_PATH = Path(__file__).resolve().parents[3] / 'design/abstention/repo/main/retrieval/generation.py'
_spec = importlib.util.spec_from_file_location('f6_generation', F6_PATH)
f6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f6)


def f6_slot(query):
    q = query.casefold().replace('’', "'")
    return next((name for name, pattern, _ in f6._SLOTS if pattern.search(q)), None)


def decide(rule, ctx):
    if rule == 'none':
        return None
    if rule == 'f6':
        return f6.abstention_reason(ctx, f6.ABSTAIN_THRESHOLD)
    if rule == 'prior':
        slot = f6_slot(ctx['query'])
        return f'prior_{slot}' if slot else None
    if rule == 'v2':
        reason = v2.abstention_reason(ctx)
        response = v2.answer(ctx, generator='extractive', abstain_threshold=v2.ABSTAIN_THRESHOLD)
        assert response['abstain'] == bool(reason), (ctx['query'], reason)
        return reason
    raise ValueError(rule)


def evaluate(rows, rule):
    out = []
    for r in rows:
        reason = decide(rule, r['context'])
        abstain = bool(reason)
        grounded = None
        if 'correct_grounded_noabstain' in r and not r.get('computed'):
            grounded = float(abstain) if r['unanswerable'] else float((not abstain) and r['correct_grounded_noabstain'] == 1)
        elif 'answer_in_gold_context' in r:   # external sets: answer string in a gold source that reached the context
            grounded = float(abstain) if r['unanswerable'] else float((not abstain) and bool(r['answer_in_gold_context']))
        out.append({**{k: r.get(k) for k in ('query_id', 'query', 'query_type', 'unanswerable', 'gold_in_context')},
                    'abstain': abstain, 'reason': reason, 'grounded': grounded})
    return out


def metrics(out):
    A = [r for r in out if not r['unanswerable']]
    U = [r for r in out if r['unanswerable']]
    aa = sum(not r['abstain'] for r in A)
    ur = sum(r['abstain'] for r in U)
    acc = aa / len(A) if A else float('nan')
    ref = ur / len(U) if U else float('nan')
    abst = sum(r['abstain'] for r in out)
    G = [r['grounded'] for r in out if r['grounded'] is not None]
    gic = [r for r in A if r['gold_in_context']]
    return {'n_answerable': len(A), 'accepted': aa, 'n_unanswerable': len(U), 'refused': ur,
            'accept': acc, 'refuse': ref, 'balanced': (acc + ref) / 2, 'abstentions': abst,
            'precision': ur / abst if abst else None,
            'grounded': sum(G) / len(G) if G else None, 'n_grounded': len(G),
            'accept_gold_in_ctx': (sum(not r['abstain'] for r in gic), len(gic))}


def fmt(name, m):
    p = 'n/a' if m['precision'] is None else f"{m['precision']:.1%}"
    g = 'n/a' if m['grounded'] is None else f"{m['grounded']:.1%} (n={m['n_grounded']})"
    return (f"{name:6s} accept {m['accepted']}/{m['n_answerable']} ({m['accept']:.1%})  refuse {m['refused']}/{m['n_unanswerable']}"
            f" ({m['refuse']:.1%})  balanced {m['balanced']:.1%}  abst-precision {p}  grounded {g}"
            f"  accept|gold-in-ctx {m['accept_gold_in_ctx'][0]}/{m['accept_gold_in_ctx'][1]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('contexts', nargs='+')
    parser.add_argument('--rules', default='none,f6,prior,v2')
    parser.add_argument('--show', default='')
    parser.add_argument('--reasons', default='')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    report = {}
    for path in args.contexts:
        rows = json.load(open(path))
        print(f'== {path} ({len(rows)} rows)')
        report[path] = {}
        for rule in args.rules.split(','):
            out = evaluate(rows, rule)
            m = metrics(out)
            report[path][rule] = m
            print(' ', fmt(rule, m))
            if rule in args.reasons.split(','):
                import collections
                print('     reasons answerable  ', dict(collections.Counter(r['reason'] for r in out if r['abstain'] and not r['unanswerable'])))
                print('     reasons unanswerable', dict(collections.Counter(r['reason'] for r in out if r['abstain'] and r['unanswerable'])))
            if rule in args.show.split(','):
                for r in out:
                    if r['abstain'] != r['unanswerable']:
                        print('     ', 'FALSE-ABSTAIN' if r['abstain'] else 'MISSED', r['reason'], '|', r['query_type'], '|', r['query'])
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
