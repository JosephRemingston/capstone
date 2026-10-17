"""Dev-only ablations of the v2 evidence set and the optional logit floor.
  cd repo && PYTHONPATH=. python ../tools/variants.py ../out/ctx_bench_dev.json ../out/ctx_calib.json"""
import json, sys, types, math
from pathlib import Path
src = Path('main/retrieval/generation.py').read_text()
KEYS = "    keys = [key for key in scored if key == best or scored[key] >= 0]\n"
SPK = "    if asked:\n        keys = [key for key in keys if not _speakers(texts[key]) or _speakers(texts[key]) & asked]\n"
assert KEYS in src and SPK in src
variants = {
    'v2 (best + logit>=0, speaker filter)': src,
    'best only': src.replace(KEYS, "    keys = [best]\n"),
    'all context sources': src.replace(KEYS, "    keys = list(scored)\n"),
    'top-2 by logit': src.replace(KEYS, "    keys = sorted(scored, key=scored.get, reverse=True)[:2]\n"),
    'logit>=-2 or best': src.replace(KEYS, "    keys = [key for key in scored if key == best or scored[key] >= -2]\n"),
    'no speaker filter': src.replace(SPK, ''),
}
rows = {p: json.load(open(p)) for p in sys.argv[1:]}
def run(mod, tau=0.0):
    out = []
    for p, rs in rows.items():
        A = [r for r in rs if not r['unanswerable']]; U = [r for r in rs if r['unanswerable']]
        acc = sum(not mod.abstention_reason(r['context'], tau) for r in A); ref = sum(bool(mod.abstention_reason(r['context'], tau)) for r in U)
        out.append(f"{Path(p).stem}: {acc}/{len(A)} acc, {ref}/{len(U)} ref, bal {(acc/len(A)+ref/len(U))/2:.1%}")
    return ' | '.join(out)
for name, code in variants.items():
    mod = types.ModuleType('g'); exec(compile(code, 'g', 'exec'), mod.__dict__)
    print(f'{name:40s}', run(mod))
mod = types.ModuleType('g'); exec(compile(src, 'g', 'exec'), mod.__dict__)
# logit floor sweep (sigmoid tau) on top of v2
logits = sorted({max(v['reranker'] for v in r['context']['scores'].values() if v['reranker'] is not None) for rs in rows.values() for r in rs})
for x in [-9, -8.5, -8, -7, -6, -5, -4, -3]:
    print(f'floor logit {x:5}', run(mod, 1 / (1 + math.exp(-x))))
