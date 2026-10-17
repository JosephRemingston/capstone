"""Compare verifier re-collected contexts to the designer's saved held-out contexts, row by row."""
import json, sys
from main.retrieval import generation as g
mine = json.load(open(sys.argv[1])); theirs = {r['query_id']: r for r in json.load(open(sys.argv[2]))}
same_src = same_logit = same_dec = 0; maxdiff = 0.0
for r in mine:
    t = theirs[r['query_id']]
    a, b = r['context'], t['context']
    same_src += list(a['sources']) == list(b['sources'])
    if set(a['sources']) == set(b['sources']):
        d = max(abs(a['scores'][k]['reranker'] - b['scores'][k]['reranker']) for k in a['sources'])
        maxdiff = max(maxdiff, d); same_logit += d < 1e-3
    same_dec += g.abstention_reason(a) == g.abstention_reason(b)
n = len(mine)
print(f'rows {n}: identical source order {same_src}/{n}, logits within 1e-3 {same_logit}/{n} (max diff {maxdiff:.2e}), same v2 decision {same_dec}/{n}')
