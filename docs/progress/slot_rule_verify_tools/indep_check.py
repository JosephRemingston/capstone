"""Independent label + metric check of the designer's saved contexts (verifier-written)."""
import json, sys, collections
from pathlib import Path
from main.retrieval import generation as g
D = Path(sys.argv[1])
def m(rows, fn):
    A=[r for r in rows if not r['unanswerable']]; U=[r for r in rows if r['unanswerable']]
    acc=sum(not fn(r) for r in A); ref=sum(bool(fn(r)) for r in U); ab=sum(bool(fn(r)) for r in rows)
    return f"acc {acc}/{len(A)} ({acc/len(A):.1%}) ref {ref}/{len(U)} ({ref/len(U):.1%}) bal {(acc/len(A)+ref/len(U))/2:.2%} prec {ref/ab if ab else float('nan'):.1%}"
v2 = lambda r: g.answer(r['context'], generator='extractive', abstain_threshold=0.0)['abstain']
# labels against raw data
lme = json.load(open('data/evaluation/longmemeval/oracle.json'))
ctx = json.load(open(D/'heldout/ctx_lme.json'))
assert [r['question_id'] for r in lme] == [r['query_id'] for r in ctx]
assert all((r['question_id'].endswith('_abs')) == c['unanswerable'] for r, c in zip(lme, ctx))
assert all(r['question'] == c['query'] == c['context']['query'] for r, c in zip(lme, ctx))
print('LME labels ok; n_abs', sum(c['unanswerable'] for c in ctx), m(ctx, v2))
loc = json.load(open('data/evaluation/locomo/locomo10.json'))
qa = [(str(r['sample_id']), i, q) for r in loc for i, q in enumerate(r['qa'])]
ctx = json.load(open(D/'heldout/ctx_locomo.json'))
assert [f'{o}:{i}' for o, i, _ in qa] == [c['query_id'] for c in ctx]
assert all((q['category'] == 5) == c['unanswerable'] and q['question'] == c['query'] for (_, _, q), c in zip(qa, ctx))
print('LoCoMo labels ok; n_cat5', sum(c['unanswerable'] for c in ctx), m(ctx, v2))
gold = json.load(open('evaluation/datasets/gold_queries.json'))
qs = gold['queries'] if isinstance(gold, dict) else gold
for name, split in (('out/ctx_bench_dev.json','dev'), ('heldout/ctx_bench_test.json','test')):
    ctx = json.load(open(D/name))
    byid = {q['id'] if 'id' in q else q['query_id']: q for q in qs}
    lab = [ (byid[c['query_id']].get('query_type')=='unanswerable' or bool(byid[c['query_id']].get('unanswerable'))) == c['unanswerable'] for c in ctx]
    print(name, 'labels match', all(lab), 'n', len(ctx), 'nU', sum(c['unanswerable'] for c in ctx), m(ctx, v2))
# status/score sanity
for name in ('heldout/ctx_lme.json','heldout/ctx_locomo.json','heldout/ctx_bench_test.json'):
    ctx = json.load(open(D/name))
    print(name, collections.Counter(c['context']['reranker']['status'] for c in ctx),
          'all scored', all(all(isinstance(c['context']['scores'][k]['reranker'], float) for k in c['context']['sources']) for c in ctx))
