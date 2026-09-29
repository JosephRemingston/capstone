"""Build the frozen 240-query CogniMem graph ablation dataset.

The labels are developer-authored from explicit templates. They are suitable for
reproducible ablations and regression testing, but are not independent human gold.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

QUERY_TYPES = ('semantic', 'episodic', 'temporal', 'historical', 'multi_hop',
               'entity_relationship', 'conflict', 'preference', 'procedural',
               'recency', 'negative_absence', 'unanswerable')

PEOPLE = ('Asha', 'Ben', 'Clara', 'Diego', 'Elena', 'Farah', 'Gavin', 'Hana',
          'Iris', 'Jonah', 'Kavya', 'Liam', 'Mina', 'Noah', 'Olivia', 'Pavel',
          'Quinn', 'Rina', 'Samir', 'Tara')
CITIES = ('Chennai', 'Lisbon', 'Kyoto', 'Nairobi', 'Oslo', 'Pune', 'Quito',
          'Rome', 'Seoul', 'Toronto', 'Utrecht', 'Vienna', 'Warsaw', 'Xiamen',
          'York', 'Zurich', 'Boston', 'Dublin', 'Helsinki', 'Jaipur', 'Bengaluru')
COMPANIES = ('Aster Labs', 'Beacon Works', 'Cedar Systems', 'Delta Studio',
             'Elm Analytics', 'Fjord Health', 'Grove Robotics', 'Harbor Cloud',
             'Ion Design', 'Juniper Media', 'Kite Energy', 'Lumen Foods',
             'Mosaic Finance', 'Nova Learning', 'Orbit Transit', 'Pine Security',
             'Quartz Bio', 'River Games', 'Solar Legal', 'Terra Maps')
ROLES = ('architect', 'chemist', 'designer', 'engineer', 'financial analyst',
         'geologist', 'historian', 'illustrator', 'journalist', 'kinesiologist',
         'librarian', 'mechanic', 'nurse', 'optician', 'planner', 'researcher',
         'statistician', 'teacher', 'urbanist', 'veterinarian')
PREFERENCES = ('oolong tea', 'quiet hotel rooms', 'morning meetings', 'dark mode',
               'window seats', 'concise summaries', 'vegetarian lunches',
               'walking routes', 'paper notebooks', 'weekly checklists',
               'decaf coffee', 'aisle seats', 'metric units', 'email updates',
               'early flights', 'audio books', 'spicy food', 'train travel',
               'blue themes', 'calendar reminders')
ALLERGENS = ('peanuts', 'shellfish', 'soy', 'sesame', 'latex', 'pollen', 'dairy',
             'gluten', 'almonds', 'eggs', 'mango', 'kiwi', 'wool', 'dust',
             'nickel', 'cats', 'penicillin', 'mustard', 'oats', 'coconut')


def _time(value: str) -> str:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).isoformat()


def _memory(identifier, content, *, scenario, created, category='semantic', entities=(), relations=()):
    return {'id': identifier, 'content': content, 'user_id': 'gold-user',
            'session_id': f'scenario-{scenario:02d}', 'category': category,
            'tier': 'long_term', 'created_at': _time(created), 'updated_at': _time(created),
            'recorded_at': _time(created), 'confidence': 1.0, 'importance_score': .8,
            'source_metadata': {'speaker': PEOPLE[scenario], 'entities': list(entities),
                                'relations': list(relations)}}


def _entity(identifier, kind, label):
    return {'id': identifier, 'type': kind, 'label': label}


def _relation(subject, predicate, target, *, start=None, end=None, positive=True):
    value = {'subject': subject, 'predicate': predicate, 'object': target, 'positive': positive}
    if start:
        value['valid_from'] = _time(start)
    if end:
        value['valid_to'] = _time(end)
    return value


def build():
    memories, queries = [], []
    for i, person in enumerate(PEOPLE):
        prefix = f's{i:02d}'
        old_city, city = CITIES[i], CITIES[i + 1]
        company, role, preference, allergen = COMPANIES[i], ROLES[i], PREFERENCES[i], ALLERGENS[i]
        manager = PEOPLE[(i + 3) % len(PEOPLE)]
        project = f'Project {chr(65 + i)}{i + 1}'
        company_id, old_city_id, city_id = f'org:{prefix}', f'location:{prefix}:old', f'location:{prefix}:new'
        manager_id, role_id = f'person:{prefix}:manager', f'concept:{prefix}:role'
        preference_id, allergen_id = f'concept:{prefix}:preference', f'concept:{prefix}:allergen'
        ids = {name: f'{prefix}-{name}' for name in ('occupation', 'event', 'old_home', 'new_home',
               'employer', 'headquarters', 'manager', 'preference', 'procedure', 'old_update',
               'new_update', 'negative')}
        memories.extend([
            _memory(ids['occupation'], f'{person} has the occupation {role}.', scenario=i, created='2024-01-10T09:00:00',
                    entities=[_entity(role_id, 'concept', role)], relations=[_relation('self', 'occupation', role_id)]),
            _memory(ids['event'], f'{person} completed the Atlas certification milestone.', scenario=i,
                    created='2024-03-12T10:00:00', category='episodic'),
            _memory(ids['old_home'], f'During 2023, the home base recorded for {person} was {old_city}.', scenario=i,
                    created='2023-01-01T08:00:00', entities=[_entity(old_city_id, 'location', old_city)],
                    relations=[_relation('self', 'lives_in', old_city_id, start='2023-01-01T00:00:00', end='2025-01-01T00:00:00')]),
            _memory(ids['new_home'], f'The current home base recorded for {person} is {city}.', scenario=i,
                    created='2025-01-01T08:00:00', entities=[_entity(city_id, 'location', city)],
                    relations=[_relation('self', 'lives_in', city_id, start='2025-01-01T00:00:00')]),
            _memory(ids['employer'], f'{person} works for {company}.', scenario=i, created='2024-04-01T09:00:00',
                    entities=[_entity(company_id, 'organization', company)],
                    relations=[_relation('self', 'works_at', company_id)]),
            _memory(ids['headquarters'], f'{company} has its headquarters in {city}.', scenario=i,
                    created='2024-04-02T09:00:00', entities=[_entity(company_id, 'organization', company),
                    _entity(city_id, 'location', city)], relations=[_relation(company_id, 'based_in', city_id)]),
            _memory(ids['manager'], f'{person} reports to {manager}.', scenario=i, created='2024-04-03T09:00:00',
                    entities=[_entity(manager_id, 'person', manager)], relations=[_relation('self', 'reports_to', manager_id)]),
            _memory(ids['preference'], f'{person} prefers {preference}.', scenario=i, created='2024-05-01T09:00:00',
                    category='preference', entities=[_entity(preference_id, 'concept', preference)],
                    relations=[_relation('self', 'likes', preference_id)]),
            _memory(ids['procedure'], f'{person} prepares the weekly report by collecting updates, checking figures, and sending a concise summary.',
                    scenario=i, created='2024-06-01T09:00:00', category='procedural'),
            _memory(ids['old_update'], f'The earlier status of {project} was planning.', scenario=i,
                    created='2025-05-01T09:00:00', category='episodic'),
            _memory(ids['new_update'], f'The latest status of {project} is launched.', scenario=i,
                    created='2025-08-01T09:00:00', category='episodic'),
            _memory(ids['negative'], f'{person} explicitly said they are not allergic to {allergen}.', scenario=i,
                    created='2024-07-01T09:00:00', entities=[_entity(allergen_id, 'concept', allergen)],
                    relations=[_relation('self', 'allergic_to', allergen_id, positive=False)]),
        ])

        def query(kind, text, relevant, expected, supporting=None, **extra):
            grades = {key: grade for key, grade in relevant}
            row = {'query_id': f'q-{prefix}-{kind}', 'query': text, 'query_type': kind,
                   'relevant_memory_ids': list(grades), 'graded_relevance': grades,
                   'expected_answer': expected, 'supporting_memory_ids': supporting or list(grades),
                   'unanswerable': not bool(grades), 'scenario_id': prefix}
            row.update(extra)
            queries.append(row)

        query('semantic', f'What is {person}\'s occupation?', [(ids['occupation'], 3)], role)
        query('episodic', f'What milestone did {person} complete?', [(ids['event'], 3)], 'Atlas certification')
        query('temporal', f'Where does {person} currently live?', [(ids['new_home'], 3)], city)
        query('historical', f'Where was {person}\'s home in 2023?', [(ids['old_home'], 3)], old_city,
              as_of=_time('2023-06-01T12:00:00'))
        query('multi_hop', f'Where is the company that {person} works for based?',
              [(ids['employer'], 3), (ids['headquarters'], 3)], city,
              [ids['employer'], ids['headquarters']])
        query('entity_relationship', f'Who does {person} report to?', [(ids['manager'], 3)], manager)
        query('conflict', f'Which city replaced {old_city} as {person}\'s current home?',
              [(ids['new_home'], 3), (ids['old_home'], 1)], city, [ids['new_home']])
        query('preference', f'What does {person} prefer?', [(ids['preference'], 3)], preference)
        query('procedural', f'How does {person} prepare the weekly report?', [(ids['procedure'], 3)],
              'collecting updates, checking figures, and sending a concise summary')
        query('recency', f'What is the most recent status of {project}?',
              [(ids['new_update'], 3), (ids['old_update'], 1)], 'launched', [ids['new_update']])
        query('negative_absence', f'Is {person} allergic to {allergen}?', [(ids['negative'], 3)],
              f'not allergic to {allergen}')
        query('unanswerable', f'What is the name of {person}\'s pet?', [], None)
    return {'schema_version': 1,
            'research_question': 'How much does adding temporal/relational graph retrieval improve vector-based conversational memory retrieval and RAG performance?',
            'controls': {'top_k': [5, 10, 20], 'candidate_limit': 20, 'context_characters': 10000,
                         'embedding_model': 'BAAI/bge-small-en-v1.5',
                         'reranker_model': 'Xenova/ms-marco-MiniLM-L-6-v2',
                         'importance_model': 'disabled; fixed record metadata is identical in every arm',
                         'generator': 'extractive by default; optional fixed Gemini configuration'},
            'label_provenance': 'developer-authored deterministic templates; not independent human gold',
            'query_types': list(QUERY_TYPES), 'memories': memories, 'queries': queries}


def main():
    target = Path(__file__).with_name('gold_queries.json')
    target.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + '\n')
    print(f'wrote {len(build()["queries"])} queries to {target}')


if __name__ == '__main__':
    main()
