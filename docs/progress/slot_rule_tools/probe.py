"""Synthetic phrasing probe written by the designer (not a benchmark): python ../tools/probe.py"""
from main.retrieval.generation import abstention_reason, question_slot

def ctx(q, texts, speaker='user'):
    return {'query': q, 'reranker': {'status': 'applied'},
            'sources': {f'm{i}': {'text': f'speaker: {speaker}\nsession_date: 2023/05/01\n{t}'} for i, t in enumerate(texts)},
            'scores': {f'm{i}': {'reranker': 3.0 - i * 4} for i in range(len(texts))}}

ANSWERABLE = [
    ('What is the name of my cat?', ["I got a kitten last month. Her name is Luna and she's a menace."]),
    ('What did I name my new puppy?', ["We finally brought the puppy home, we're calling him Max."]),
    ('What was the name of the restaurant I went to for my birthday?', ["For my birthday we went to Nopa downtown, the pork chop was amazing."]),
    ("What is my sister's name?", ["My sister, Rachel, is visiting next week."]),
    ("What's the name of my manager?", ["I had a 1:1 with my manager Tom about the promotion."]),
    ('What does my wife do for a living?', ["My wife is a high school math teacher so summers are quiet."]),
    ('What does my brother do?', ["My brother just got a job as a software developer in Austin."]),
    ('What is my current job title?', ["I was promoted to senior product manager in March."]),
    ('What do I do for work?', ["I work as a nurse on night shifts, so sleep is tricky."]),
    ('Which city did I move to last year?', ["Moving to Denver last year was the best decision."]),
    ('Which university did I graduate from?', ["I graduated from UCLA with a degree in economics."]),
    ('Which company did I start working at in June?', ["I started at Stripe in June and the onboarding is intense."]),
    ('How often do I go to the gym?', ["I hit the gym three times a week, usually mornings."]),
    ('How often do I call my mom?', ["I call my mom every Sunday evening."]),
    ('How frequently do I water the plants?', ["The plants get watered twice a week."]),
    ('How much did I pay for my new laptop?', ["The new laptop was $1,200 which hurt."]),
    ('How much is my monthly rent?', ["Rent is 1500 a month for the one-bedroom, plus utilities. That's $1500."]),
    ('How long is my daily commute?', ["My commute is about 45 minutes each way."]),
    ('How long did my trip to Japan last?', ["I was in Japan for two weeks in April."]),
    ('How long do I bake the bread?', ["Bake the loaf at 230C for 40 minutes."]),
    ('How many hours do I sleep on weeknights?', ["On weeknights I only get about six hours of sleep."]),
    ('How far is my office from home?', ["The office is 12 km from my place."]),
    ('How far did I run on Saturday?', ["Saturday I ran 10 miles along the river."]),
    ('What is the name of the yoga studio I joined?', ["I joined CorePower Yoga near work."]),
    ("What's my dog's name?", ["Biscuit, my dog, refuses to walk in the rain."]),
    ('How often does Caroline paint?', ["I paint almost every weekend these days."]),
    ('What does Melanie do for work?', ["I'm a counselor at a youth center."]),
    ('How long have I been learning Spanish?', ["I started Spanish lessons in 2021."]),
    ('How many days did it take for my package to arrive after I ordered it?', ["The package arrived on Friday."]),
    ('How much did I spend on gifts in total?', ["I spent about $50 on gifts for Ana."]),
    ('Is my dog called Biscuit?', ["Our dog Biscuit loves the beach."]),
    ('Did I ever meet someone named Carlos?', ["I met Carlos at the conference."]),
    ('How much do I like jazz?', ["I love jazz, especially Coltrane."]),
    ("What is John's number one goal in his basketball career?", ["My number one goal is to win a championship."]),
    ('What do I usually order at Starbucks?', ["I usually get an oat latte."]),
    ('What did my friend recommend for the trip?', ["My friend Sam recommended the Lisbon food tour."]),
    ("What's the name of the book I'm reading?", ["Currently reading The Overstory and loving it."]),
    ('What was the name of the hotel we stayed at in Rome?', ["We stayed at Hotel Artemide near Termini."]),
    ('How often did I visit the dentist last year?', ["I went to the dentist twice last year."]),
    ('How long was the flight to Tokyo?', ["The flight to Tokyo was almost 14 hours."]),
]
UNANSWERABLE = [
    ('What is the name of my cat?', ["Adopted a cat last week and she already rules the house."]),
    ('What is the name of my cat?', ["Yesterday the cat knocked over my plant. Typical Monday."]),
    ('What does my brother do for a living?', ["My brother lives in Leeds and loves football."]),
    ('What does my brother do for a living?', ["My brother is a vegetarian, so no BBQ."]),
    ("What is my sister's name?", ["My sister's wedding is in Goa this year."]),
    ('How often do I go swimming?', ["I love swimming in the sea."]),
    ('How much did I pay for the concert tickets?', ["The concert tickets sold out quickly."]),
    ('How long is my commute?', ["I take the metro to work, it is crowded."]),
    ('Which city does my aunt live in?', ["My aunt is coming over for dinner on Friday."]),
    ('Which university did I go to?', ["I studied economics and loved it."]),
]
fa = [(q, abstention_reason(ctx(q, t))) for q, t in ANSWERABLE]
mr = [(q, abstention_reason(ctx(q, t))) for q, t in UNANSWERABLE]
print('answerable false abstentions', sum(bool(r) for _, r in fa), '/', len(fa))
for (q, r), (_, t) in zip(fa, ANSWERABLE):
    print('  ', 'ABSTAIN' if r else 'answer ', question_slot(q), r or '', '|', q)
print('unanswerable refused', sum(bool(r) for _, r in mr), '/', len(mr))
for q, r in mr:
    print('  ', 'REFUSE ' if r else 'answer ', question_slot(q), '|', q)
