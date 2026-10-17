"""Show the v2 slot decision for each slot question in context files: python ../tools/explain.py FILE... [--all]"""
import json, sys
from main.retrieval import generation as g
for path in [a for a in sys.argv[1:] if not a.startswith('--')]:
    for r in json.load(open(path)):
        ctx = r['context']
        ev = g._slot_evidence(ctx)
        if not ev or ev[0] is None:
            continue
        slot, sentences, excluded, mid = ev
        reason = g.abstention_reason(ctx)
        wrong = bool(reason) != r['unanswerable']
        if not wrong and '--all' not in sys.argv:
            continue
        print(('WRONG ' if wrong else 'ok    ') + ('U' if r['unanswerable'] else 'A'), slot, reason, '|', r['query'], '| ans:', r.get('expected_answer'))
        for s in sentences:
            fill = g._names(s, mid[0], excluded, mid[2]) if slot in ('name', 'entity') else ''
            print('      -', s[:150], '=>', fill)
