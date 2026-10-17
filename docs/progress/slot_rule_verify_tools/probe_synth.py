"""Verifier's adversarial probes on hand-built contexts (rule is a pure function of the context).
Each source: (text, logit). The best-logit source is always in the evidence."""
from main.retrieval.generation import abstention_reason, question_slot

def ctx(q, items, speaker='user'):
    return {'query': q, 'reranker': {'status': 'applied'},
            'sources': {f'm{i}': {'text': f'speaker: {speaker}\nsession_date: 2024/03/01\n{t}'} for i, (t, _) in enumerate(items)},
            'scores': {f'm{i}': {'reranker': s} for i, (_, s) in enumerate(items)}}

A = [  # answerable: the answer IS in the evidence -> must NOT abstain
 ('A01', "What's my dog's name?", [("Rex is my dog. He hates the rain.", 4.0)]),
 ('A02', "What is my sister's name?", [("Ana and I went to the beach; she's my sister and my best friend.", 3.0)]),
 ('A03', "What does my mum do?", [("Mum teaches piano to kids at the community centre.", 2.0)]),
 ('A04', "What does my brother do for a living?", [("My brother codes for a bank in Frankfurt.", 2.0)]),
 ('A05', "How long did I live in Berlin?", [("I lived in Berlin from 2015 to 2019.", 5.0)]),
 ('A06', "How much did my bike cost?", [("The new bike was 800, way over budget.", 4.0)]),
 ('A07', "How often do I play tennis?", [("I play tennis biweekly with Sam.", 4.0)]),
 ('A08', "What's the name of the bakery on Elm Street?", [("The bakery on Elm Street is literally called Elm Street Bakery.", 5.0)]),
 ('A09', "What is my cat called?", [("my cat is called mochi and she is tiny.", 5.0)]),
 ('A10', "Which city did I move to?", [("i moved to lisbon last spring for work.", 5.0)]),
 ('A11', "What does my partner do?", [("My partner is in IT, mostly cloud stuff.", 3.0)]),
 ('A12', "What is my boss's name?", [("I had a tough meeting today.", 1.0), ("Dr. Okafor, who manages our team, approved my leave.", -0.5)]),
 ('A13', "How far is my gym?", [("The gym is a short walk, maybe five minutes.", 4.0)]),
 ('A14', "Which school does my son go to?", [("My son started at St. Mary's in September.", 5.0)]),
 ('A15', "What's the name of my therapist?", [("My therapist, Joanna, suggested journaling.", 3.0)]),
 ('A16', "How often do I visit my grandparents?", [("I visit my grandparents every other month.", 3.0)]),
 ('A17', "How long is my commute?", [("My commute is forty five minutes on a good day.", 4.0)]),
 ('A18', "What was the name of the hotel in Kyoto?", [("We stayed at the hotel in Kyoto that Mia recommended, the Ace.", 3.0)]),
 ('A19', "How much is my rent?", [("Rent went up to 1,450 a month.", 4.0)]),
 ('A20', "What do my parents call their boat?", [("My parents named their boat 'sea breeze'.", 4.0)]),
 ('A21', "Which company does my wife work for?", [("My wife works for a startup called Lumen.", 4.0)]),
 ('A22', "How long was my flight to Lima?", [("The flight to Lima took 11 hours.", 4.0)]),
 ('A23', "What does Melanie do for work?", [("I'm a counselor at a youth center.", 4.0)], 'Melanie'),
 ('A24', "What is the name of my son's teacher?", [("Mrs. Patel, his teacher, says he is doing great.", 4.0)]),
 ('A25', "How often does Jon go to the gym?", [("I go to the gym four days a week.", 4.0)], 'Jon'),
]
U = [  # unanswerable: the asked-for attribute is NOT in the evidence -> should abstain
 ('U01', "What is my sister's name?", [("My sister lives in Madrid now.", 4.0)]),
 ('U02', "What's my dog's name?", [("I took my dog to Central Park this morning.", 4.0)]),
 ('U03', "What's the name of the restaurant where I had my birthday dinner?", [("For my birthday dinner we went to a little Italian place downtown.", 4.0)]),
 ('U04', "Which country did I visit in June?", [("In June I tried Thai food for the first time.", 4.0)]),
 ('U05', "How long is my commute?", [("My commute starts at 8:30 every day.", 4.0)]),
 ('U06', "How often do I go running?", [("I always forget my headphones when running.", 4.0)]),
 ('U07', "What does my brother do for a living?", [("My brother is a huge fan of the director Nolan.", 4.0)]),
 ('U08', "What does my sister do for a living?", [("My sister is a keen baker on weekends.", 4.0)]),
 ('U09', "Which city did my parents move to?", [("My parents moved in May with help from Uncle Joe.", 4.0)]),
 ('U10', "How much did I pay for the flight?", [("The flight was free of turbulence, thankfully.", 4.0)]),
 ('U11', "Who is my dentist?", [("I need to book a dentist appointment soon.", 4.0)]),
 ('U12', "When is my mother's birthday?", [("My mother loves gardening.", 4.0)]),
 ('U13', "What is my wife's name?", [("My wife and I watched Dune on Friday.", 4.0)]),
 ('U14', "Which university did my brother attend?", [("My brother studied engineering. Go Bears!", 4.0)]),
 ('U15', "What is my cat's name?", [("The cat sleeps on my laptop. Typical Monday.", 4.0)]),
 ('U16', "How much did my sofa cost?", [("The sofa took 3 weeks to arrive.", 4.0)]),
 ('U17', "How long did my trip to Peru last?", [("My trip to Peru was in 2019.", 4.0)]),
 ('U18', "What's my landlord's name?", [("My landlord texted me on WhatsApp about the boiler.", 4.0)]),
 ('U19', "What is the name of my dog?", [("Rex is my neighbour's dog, I love him.", 4.0), ("I want a dog of my own someday.", 1.0)]),
 ('U20', "How often do I water the plants?", [("I watered the plants on Monday.", 4.0)]),
 ('U21', "What does my uncle do?", [("My uncle lives in Accra with his wife, a nurse.", 4.0)]),
 ('U22', "Which city does my friend Tom live in?", [("Tom and I met at a conference in Berlin.", 4.0)]),
]
def run(rows, want):
    bad = 0
    for row in rows:
        rid, q, items = row[:3]; spk = row[3] if len(row) > 3 else 'user'
        r = abstention_reason(ctx(q, items, spk))
        ok = bool(r) == want
        bad += not ok
        print(f"  {rid} {'ok ' if ok else 'BAD'} slot={question_slot(q)!s:10} reason={r!s:28} | {q}")
    return bad
print('ANSWERABLE (should answer)'); fa = run(A, False)
print('UNANSWERABLE (should abstain)'); mr = run(U, True)
print(f'false abstentions {fa}/{len(A)}; refused {len(U)-mr}/{len(U)}; total probes {len(A)+len(U)}')
