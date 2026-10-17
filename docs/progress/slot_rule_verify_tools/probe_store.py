"""Verifier probe over a small real store: full hybrid retrieval + local reranker, then the v2 rule.
Memories and questions written by the verifier (not taken from any benchmark)."""
from datetime import datetime, timezone, timedelta
import json, sys, tempfile
from pathlib import Path
from main.domain.models import MemoryCategory, MemoryRecord, MemoryTier
from main.retrieval.embeddings import FastEmbedder
from main.retrieval.generation import abstention_reason, question_slot
from evaluation.locomo import SnapshotStore
from evaluation.systems import CachingEmbedder, Protocol, build_rag

MEM = [
 "My sister Leona just started her residency in Glasgow, she's going to be a paediatrician.",
 "Our dog is a scruffy terrier, we call him Pickle.",
 "I moved to Rotterdam in 2022 after the Utrecht job fell through.",
 "I work at Vandermolen Logistics as a data analyst, mostly forecasting freight volumes.",
 "I go climbing at the bouldering gym three evenings a week.",
 "Rent for the flat is €1,350 a month, which is steep for two rooms.",
 "My commute is about 25 minutes by bike if the wind isn't brutal.",
 "My dad retired last year; he used to be a ship's engineer for 30 years.",
 "Mum runs a small florist shop in Leiden called Bloemkast.",
 "I'm learning Portuguese on Duolingo, I practise every morning before work.",
 "My best friend Tariq is getting married in Lisbon next June.",
 "I paid 89 euros for the new climbing shoes, La Sportiva ones.",
 "The marathon I ran in Amsterdam took me 4 hours and 12 minutes.",
 "I usually call my grandma on Sunday afternoons.",
 "My partner Jun is a sound engineer and works on podcasts from home.",
 "I'm allergic to cashews, found out the hard way at a wedding.",
 "We adopted a second cat in March and named her Olive.",
 "Our first cat, Miso, is 9 and very grumpy about Olive.",
 "I studied civil engineering at TU Delft before switching to data.",
 "I volunteer at the food bank twice a month on Saturdays.",
 "My brother lives in Antwerp and just bought a house with his girlfriend.",
 "I want to get my sailing licence this summer.",
 "The Rotterdam half marathon is on my list for next spring.",
 "I'm trying to cut down on coffee, now down to two cups a day.",
 "My manager, Ingrid Sato, approved my four-day week starting in May.",
 "We went to Porto for a long weekend in October and loved the tiles everywhere.",
 "I play five-a-side football with colleagues on Thursday nights.",
 "My cousin Emre is studying medicine in Izmir.",
 "My neighbour keeps parking his van in front of our bike shed.",
 "I've been reading a lot of Ursula Le Guin lately, The Dispossessed was amazing.",
]
A = [  # (question, should_answer=True)
 "What's my sister's name?", "What is my sister going to be?", "What's our dog called?", "Which city did I move to in 2022?",
 "Which company do I work for?", "What do I do for work?", "How often do I go climbing?", "How much is my rent?",
 "How long is my commute?", "What did my dad do before he retired?", "What's the name of my mum's shop?",
 "What does my mum do?", "How often do I practise Portuguese?", "Where is Tariq getting married?",
 "How much did my climbing shoes cost?", "How long did the Amsterdam marathon take me?", "How often do I call my grandma?",
 "What does Jun do?", "What's my partner's name?", "What is the name of the cat we adopted in March?", "How old is Miso?",
 "Which university did I go to?", "How often do I volunteer at the food bank?", "How many cups of coffee do I drink a day?",
 "What is my manager's name?", "Which city did we visit in October?", "How often do I play football?",
 "What is my cousin studying?", "What's my best friend called?", "What am I allergic to?",
]
U = [  # should abstain
 "What's my brother's name?", "What does my brother do for a living?", "What is my girlfriend's name?",
 "How much did my brother's house cost?", "How often do I go sailing?", "What's my neighbour's name?",
 "Which city does my cousin Emre live in now?", "How long have I had Pickle?", "What's the name of my brother's girlfriend?",
 "How much did the Porto trip cost?", "What is my grandma's name?", "Which school did my sister go to?",
 "How far is the bouldering gym from my flat?", "What does my neighbour do?", "How much do I earn?",
 "What is the name of the bouldering gym I go to?", "How long was my Porto flight?", "What's the name of my football team?",
 "What does Tariq do for a living?", "Which hospital does Leona work at?",
]
def main():
    when = datetime(2025, 1, 1, tzinfo=timezone.utc)
    recs = [MemoryRecord(id=f'p{i:02d}', content=t, user_id='probe', session_id=f's{i}', category=MemoryCategory.SEMANTIC,
                         tier=MemoryTier.LONG_TERM, created_at=when + timedelta(days=i), updated_at=when + timedelta(days=i),
                         recorded_at=when + timedelta(days=i), source_metadata={'speaker': 'user'}) for i, t in enumerate(MEM)]
    out = Path(sys.argv[1]); out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(dir=out.parent) as d:
        store = SnapshotStore(Path(d) / 'p.jsonl', recs)
        rag = build_rag(store, embedder=CachingEmbedder(FastEmbedder(), A + U), protocol=Protocol(candidate_limit=20))
        rag.sync(user_id='probe')
        for q, want_answer in [(q, True) for q in A] + [(q, False) for q in U]:
            ctx = rag.context(q, user_id='probe', budget=10000, limit=5, candidate_limit=20, mode='hybrid',
                              channels=('keyword', 'semantic', 'graph'), rerank=True)
            r = abstention_reason(ctx, 0.0)
            ok = (r is None) == want_answer
            top = max(ctx['sources'], key=lambda k: ctx['scores'][k]['reranker'] or -99)
            rows.append({'q': q, 'answerable': want_answer, 'slot': question_slot(q), 'reason': r, 'ok': ok,
                         'status': ctx['reranker']['status'], 'top': ctx['sources'][top]['text'][-110:].replace('\n', ' '),
                         'top_logit': ctx['scores'][top]['reranker']})
    out.write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(f"{'ok ' if r['ok'] else 'BAD'} {'A' if r['answerable'] else 'U'} slot={r['slot']!s:10} reason={r['reason']!s:26} {r['q']}  || top[{r['top_logit']:.1f}]: {r['top'][-70:]}")
    fa = sum(not r['ok'] for r in rows if r['answerable']); mr = sum(r['ok'] for r in rows if not r['answerable'])
    print(f"answerable false abstentions {fa}/{len(A)}; unanswerable refused {mr}/{len(U)}; statuses {set(r['status'] for r in rows)}")
main()
