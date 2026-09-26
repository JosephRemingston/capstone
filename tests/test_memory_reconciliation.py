"""Stateful checks for conflict resolution, source evidence and consolidation."""
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest

from main import MemoryCore, MemoryInput, LocalMemoryStore, MemoryRecord, MemoryTier
from main.__main__ import main as cli_main

NOW = datetime(2026, 9, 26, 10, tzinfo=timezone.utc)


def memory(text, *, offset=0, user='u', session='s', role='user', **metadata):
    return MemoryCore().process(MemoryInput(content=text, user_id=user, session_id=session,
                                           role=role, timestamp=NOW + timedelta(seconds=offset), metadata=metadata))


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = LocalMemoryStore(Path(directory.name) / 'store.jsonl')

    def add(self, text, **kwargs):
        return self.store.ingest(memory(text, **kwargs))

    def test_city_conflict_preserves_history_and_default_retrieval_is_current(self):
        old = self.add('I live in Chennai.')
        new = self.add('I live in Bengaluru.', offset=60)
        updated = self.store.get(old.id)
        self.assertEqual(updated.memory_status, 'superseded')
        self.assertEqual(updated.superseded_by, new.id)
        self.assertEqual(updated.content, old.content)
        self.assertEqual(new.conflict_ids, [old.id])
        self.assertEqual(updated.conflict_ids, [new.id])
        self.assertEqual(new.conflict_resolution['reason'], 'observation_time')
        self.assertEqual([r.id for r in self.store.list(user_id='u', now=NOW)], [new.id])
        self.assertEqual(self.store.search('Chennai', now=NOW), [])
        self.assertEqual(self.store.search('Chennai', include_history=True, now=NOW)[0].id, old.id)
        self.assertEqual([r.memory_status for r in self.store.history(old.id, user_id='u')], ['active', 'superseded'])
        self.assertEqual(self.store.history(old.id, user_id='another'), [])
        before = self.store.path.read_bytes()
        self.store.search('Bengaluru', now=NOW)
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_compatible_preferences_coexist_but_opposite_polarities_conflict(self):
        tea = self.add('I like tea.')
        coffee = self.add('I like coffee.')
        negative = self.add("I don't like coffee.", offset=60)
        self.assertEqual(self.store.get(tea.id).memory_status, 'active')
        self.assertEqual(self.store.get(coffee.id).superseded_by, negative.id)
        self.assertEqual(len(self.store.list(now=NOW)), 2)
        favorite = self.add('My favorite drink is tea.', offset=100)
        replacement = self.add('My favourite drink is coffee.', offset=200)
        self.assertEqual(self.store.get(favorite.id).superseded_by, replacement.id)
        self.assertEqual(self.store.get(negative.id).memory_status, 'active')

    def test_preference_dimensions_and_comparative_choice(self):
        short = self.add('I prefer short answers.')
        long = self.add('I prefer detailed answers.', offset=60)
        self.assertEqual(self.store.get(short.id).superseded_by, long.id)
        tea = self.add('I prefer tea over coffee.')
        coffee = self.add('I prefer coffee over tea.', offset=60)
        self.assertEqual(self.store.get(tea.id).superseded_by, coffee.id)
        self.assertEqual(self.store.get(long.id).memory_status, 'active')

    def test_preference_changes_with_negation_now_and_indirect_wording(self):
        short = self.add('I would rather receive short answers.')
        rejected = self.add("I don't like short answers.", offset=30)
        self.assertEqual(self.store.get(short.id).superseded_by, rejected.id)
        equivalent = self.add('I dislike brief answers.', offset=60)
        self.assertEqual(equivalent.consolidated_into, rejected.id)
        self.assertEqual(self.store.get(rejected.id).summary, 'User does not prefer concise responses.')
        coffee = self.add('I like coffee.')
        negative = self.add('I no longer like coffee.', offset=30)
        self.assertEqual(self.store.get(coffee.id).superseded_by, negative.id)
        positive = self.add('I now like coffee.', offset=60)
        self.assertEqual(self.store.get(negative.id).superseded_by, positive.id)
        self.assertEqual(self.add('I like coffee now.', offset=90).consolidated_into, positive.id)

    def test_users_entities_and_attributes_stay_separate(self):
        me = self.add('I live in Chennai.')
        foreign = self.add('I live in Mumbai.', user='other', offset=60)
        alice = self.add('Alice lives in Chennai.')
        bob = self.add('Bob lives in Mumbai.', offset=60)
        name = self.add('My name is Joseph.')
        changed = self.add('Alice lives in Bengaluru.', offset=100)
        self.assertEqual(self.store.get(alice.id).superseded_by, changed.id)
        for record in (me, foreign, bob, name):
            self.assertEqual(self.store.get(record.id).memory_status, 'active')

    def test_negative_facts_do_not_invent_a_replacement_city(self):
        city = self.add('I live in Chennai.')
        not_mumbai = self.add('I do not live in Mumbai.', offset=30)
        self.assertEqual(self.store.get(city.id).memory_status, 'active')
        not_chennai = self.add('I no longer live in Chennai.', offset=60)
        self.assertEqual(self.store.get(city.id).superseded_by, not_chennai.id)
        self.assertEqual(self.store.get(not_mumbai.id).memory_status, 'active')
        self.assertFalse(not_chennai.claim['positive'])

    def test_rejected_assertion_does_not_supersede_compatible_existing_facts(self):
        chennai = self.add('I live in Chennai.', confidence=.8)
        not_mumbai = self.add('I do not live in Mumbai.', confidence=.9)
        rejected = self.add('I live in Mumbai.', confidence=.85, offset=60)
        self.assertEqual(rejected.superseded_by, not_mumbai.id)
        self.assertEqual(self.store.get(chennai.id).memory_status, 'active')
        self.assertEqual(self.store.get(not_mumbai.id).memory_status, 'active')

    def test_unsafe_and_mixed_statements_never_supersede_a_fact(self):
        city = self.add('I live in Chennai.')
        for text in ['If I live in Mumbai, I will call you.', 'I used to live in Mumbai.',
                     'I might live in Mumbai.', 'I live in Mumbai and work at Acme.',
                     'She said "I live in Mumbai".', 'Do I live in Mumbai?',
                     'I do not dislike Mumbai.', 'I live in Mumbai during summer.']:
            with self.subTest(text=text):
                incoming = self.add(text, offset=60)
                self.assertIsNone(incoming.claim)
                self.assertEqual(self.store.get(city.id).memory_status, 'active')
        self.add('I live in Mumbai.', offset=100, role='assistant')
        self.assertEqual(self.store.get(city.id).memory_status, 'active')

    def test_resolution_priority_confidence_time_importance_and_ties(self):
        for field, left_kwargs, right_kwargs, reason in [
            ('priority', {'source_priority': 90}, {'source_priority': 10}, 'source_priority'),
            ('confidence', {'confidence': .9}, {'confidence': .2}, 'confidence'),
            ('time', {'offset': 100}, {'offset': 50}, 'observation_time'),
        ]:
            with self.subTest(field=field):
                user = field
                left = self.add('I live in Chennai.', user=user, **left_kwargs)
                right = self.add('I live in Mumbai.', user=user, **right_kwargs)
                self.assertEqual(right.memory_status, 'superseded')
                self.assertEqual(right.superseded_by, left.id)
                self.assertEqual(right.conflict_resolution['reason'], reason)
        left = self.store.ingest(replace(memory('I live in Chennai.', user='importance'), importance_score=.8))
        right = self.store.ingest(replace(memory('I live in Mumbai.', user='importance'), importance_score=.6))
        self.assertEqual(right.conflict_resolution['reason'], 'importance')
        self.assertEqual(right.superseded_by, left.id)
        tie = self.store.ingest(replace(memory('I live in Delhi.', user='importance'), importance_score=.8))
        self.assertEqual(tie.superseded_by, left.id)
        self.assertEqual(tie.conflict_resolution['reason'], 'tie_kept_existing')

    def test_touches_do_not_override_observation_chronology(self):
        old = self.add('I live in Chennai.')
        old.touch(NOW + timedelta(days=7))
        self.store.save(old)
        new = self.add('I live in Mumbai.', offset=60)
        self.assertEqual(self.store.get(old.id).superseded_by, new.id)

    def test_equivalent_facts_consolidate_and_retries_do_not_add_evidence(self):
        original = self.add('I live in Chennai.')
        duplicate = self.add('I reside in Chennai.', offset=60, session='s2')
        self.assertEqual(duplicate.memory_status, 'consolidated')
        self.assertEqual(duplicate.consolidated_into, original.id)
        canonical = self.store.get(original.id)
        self.assertEqual(canonical.summary, 'User lives in Chennai.')
        self.assertEqual(set(canonical.evidence_ids), {original.id, duplicate.id})
        self.assertEqual(canonical.last_observed_at, duplicate.created_at)
        self.assertEqual(canonical.first_observed_at, original.created_at)
        self.assertEqual(len(self.store.list(now=NOW)), 1)
        self.assertEqual(self.store.list(session_id='s2', now=NOW)[0].id, original.id)
        self.assertEqual(len(self.store.list(include_history=True, now=NOW)), 2)
        before = self.store.path.read_bytes()
        self.store.ingest(duplicate)
        self.assertEqual(self.store.path.read_bytes(), before)
        self.assertEqual(MemoryRecord.from_dict(canonical.to_dict()), canonical)

    def test_current_wording_and_names_keep_assertion_identity(self):
        original = self.add('I live in Chennai.')
        for text in ['I live in Chennai now.', 'I currently reside in Chennai.', 'Currently, I live in Chennai.']:
            with self.subTest(text=text):
                self.assertEqual(self.add(text, offset=60).consolidated_into, original.id)
        cat = self.add("My cat's name is Luna.")
        changed = self.add("My cat’s name is Milo.", offset=60)
        self.assertEqual(self.store.get(cat.id).superseded_by, changed.id)
        self.assertEqual(self.store.get(original.id).memory_status, 'active')

    def test_out_of_order_duplicate_preserves_latest_support(self):
        original = self.add('I live in Chennai.', offset=100)
        self.add('I reside in Chennai.', offset=50)
        latest = self.store.get(original.id)
        self.assertEqual(latest.first_observed_at, NOW + timedelta(seconds=50))
        self.assertEqual(latest.last_observed_at, original.created_at)
        stale = self.add('I live in Mumbai.', offset=75)
        self.assertEqual(stale.superseded_by, original.id)

    def test_exact_durable_duplicates_preserve_complex_wording(self):
        for text in ['I usually take tea without milk.', 'First run tests, then deploy.',
                     'My hobbies are painting and drawing.']:
            with self.subTest(text=text):
                original = self.add(text)
                repeated = self.add(text, offset=60)
                self.assertEqual(repeated.consolidated_into, original.id)
                self.assertEqual(self.store.get(original.id).summary, text)
                self.assertEqual(len(self.store.get(original.id).evidence_ids), 2)

    def test_rank_uses_an_actual_supporting_observation(self):
        original = self.add('I live in Chennai.', confidence=.9)
        self.add('I reside in Chennai.', confidence=.1, offset=120)
        changed = self.add('I live in Mumbai.', confidence=.9, offset=60)
        self.assertEqual(self.store.get(original.id).superseded_by, changed.id)

    def test_fresh_support_updates_retention_without_count_based_score_inflation(self):
        old = self.add('I live in Chennai.', offset=-120 * 86400)
        old.importance_score = .6
        old.tier = MemoryTier.ARCHIVE
        self.store.save(old)
        fresh = self.add('I reside in Chennai.')
        root = self.store.get(old.id)
        self.assertEqual(root.tier, MemoryTier.LONG_TERM)
        self.assertEqual(root.importance_score, .6)
        self.assertEqual(root.archive_after, NOW + timedelta(days=90))
        self.assertEqual(root.evidence_ids, [old.id, fresh.id])
        self.assertEqual(self.store.list(now=NOW)[0].tier, MemoryTier.LONG_TERM)

    def test_returning_to_an_old_value_starts_new_evidence_episode(self):
        a = self.add('I live in Chennai.')
        b = self.add('I live in Mumbai.', offset=60)
        c = self.add('I live in Chennai.', offset=120)
        self.assertEqual(self.store.get(a.id).superseded_by, b.id)
        self.assertEqual(self.store.get(b.id).superseded_by, c.id)
        self.assertEqual(c.evidence_ids, [c.id])
        self.assertIsNone(c.consolidated_into)
        self.assertEqual([r.id for r in self.store.list(now=NOW)], [c.id])

    def test_manual_selection_restores_old_root_and_preserves_audit(self):
        a = self.add('I live in Chennai.')
        b = self.add('I live in Mumbai.', offset=60)
        selected = self.store.resolve_conflict(a.id, user_id='u', now=NOW + timedelta(seconds=120))
        self.assertEqual(selected.memory_status, 'active')
        self.assertIsNone(selected.superseded_by)
        self.assertEqual(selected.conflict_resolution['reason'], 'explicit_selection')
        self.assertEqual(self.store.get(b.id).superseded_by, a.id)
        self.assertEqual(len(self.store.history(a.id, user_id='u')), 3)
        before = self.store.path.read_bytes()
        with self.assertRaises(ValueError):
            self.store.resolve_conflict(b.id, user_id='other', now=NOW + timedelta(seconds=180))
        with self.assertRaises(ValueError):
            self.store.resolve_conflict(b.id, user_id='u', now=NOW)
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_manual_confirmation_is_not_undone_by_equal_quality_late_data(self):
        a = self.add('I live in Chennai.')
        self.add('I live in Mumbai.', offset=60)
        self.store.resolve_conflict(a.id, user_id='u', now=NOW + timedelta(seconds=180))
        late = self.add('I live in Delhi.', offset=120)
        self.assertEqual(late.superseded_by, a.id)
        later = self.add('I live in Kolkata.', offset=240)
        self.assertEqual(self.store.get(a.id).superseded_by, later.id)

    def test_bulk_reconciliation_of_old_records_is_idempotent_and_user_scoped(self):
        a = self.store.save(memory('I live in Chennai.'))
        duplicate = self.store.save(memory('I reside in Chennai.', offset=60))
        new = self.store.save(memory('I live in Mumbai.', offset=120))
        foreign = self.store.save(memory('I live in Delhi.', user='other'))
        before_foreign = foreign.to_dict()
        changed = self.store.reconcile_memories(user_id='u')
        self.assertEqual(len(changed), 3)
        self.assertEqual(self.store.get(a.id).superseded_by, new.id)
        self.assertEqual(self.store.get(duplicate.id).consolidated_into, a.id)
        self.assertEqual(self.store.get(foreign.id).to_dict(), before_foreign)
        before = self.store.path.read_bytes()
        self.assertEqual(self.store.reconcile_memories(user_id='u'), [])
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_bulk_keeps_existing_evidence_when_merging_legacy_duplicate(self):
        a = self.add('I live in Chennai.')
        b = self.add('I reside in Chennai.', offset=60)
        c = self.store.save(memory('I live in Chennai.', offset=120))
        self.store.reconcile_memories(user_id='u')
        self.assertEqual(set(self.store.get(a.id).evidence_ids), {a.id, b.id, c.id})
        self.assertEqual(self.store.get(c.id).consolidated_into, a.id)

    def test_tasks_and_separate_events_are_not_consolidated(self):
        a = self.add('Submit the report tomorrow.')
        b = self.add('Submit the report tomorrow.', offset=60)
        self.assertEqual(self.store.get(a.id).memory_status, 'active')
        self.assertEqual(b.memory_status, 'active')
        self.assertEqual(b.evidence_ids, [])
        self.add('I submitted the report.', task_id=a.id, offset=120)
        self.assertEqual(self.store.get(a.id).task_status, 'completed')
        self.assertEqual(self.store.get(b.id).task_status, 'active')

    def test_invalid_quality_metadata_is_rejected_before_write(self):
        original = self.add('I live in Chennai.')
        before = self.store.path.read_bytes()
        for metadata in [{'confidence': -1}, {'confidence': float('nan')}, {'source_priority': 101},
                         {'confidence': 'high'}, {'source_priority': True}]:
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                self.add('I live in Mumbai.', **metadata)
        with self.assertRaises(ValueError):
            self.store.ingest(replace(memory('I live in Mumbai.'), confidence=float('inf')))
        self.assertEqual(self.store.path.read_bytes(), before)
        self.assertEqual(self.store.get(original.id).memory_status, 'active')

    def test_imported_evidence_cannot_reference_another_user_or_claim(self):
        for index, source in enumerate([memory('I live in Chennai.', user='other'), memory('I live in Mumbai.', user='subject-1')]):
            with self.subTest(source=source.content):
                self.store.save(source)
                imported = memory('I live in Chennai.', user=f'subject-{index}')
                imported.evidence_ids = [imported.id, source.id]
                self.store.save(imported)
                before = self.store.path.read_bytes()
                with self.assertRaises(ValueError):
                    self.add('I reside in Chennai.', offset=60, user=f'subject-{index}')
                self.assertEqual(self.store.path.read_bytes(), before)

    def test_old_schema_defaults(self):
        payload = memory('I live in Chennai.').to_dict()
        for key in ('memory_status', 'claim', 'superseded_by', 'consolidated_into', 'conflict_ids',
                    'conflict_resolution', 'evidence_ids', 'evidence_session_ids', 'summary',
                    'first_observed_at', 'last_observed_at', 'last_confirmed_at'):
            payload.pop(key)
        restored = MemoryRecord.from_dict(payload)
        self.assertEqual(restored.memory_status, 'active')
        self.assertIsNone(restored.claim)
        self.store.save(restored)
        new = self.add('I live in Mumbai.', offset=60)
        self.assertEqual(self.store.get(restored.id).superseded_by, new.id)

    def test_cli_ingest_history_selection_and_reconcile(self):
        def cli(*args):
            output = io.StringIO()
            with redirect_stdout(output):
                code = cli_main(['--store', str(self.store.path), *args])
            self.assertEqual(code, 0, output.getvalue())
            return json.loads(output.getvalue())
        old = cli('process', 'I live in Chennai.', '--user-id', 'u', '--session-id', 's', '--confidence', '.8')
        new = cli('process', 'I live in Mumbai.', '--user-id', 'u', '--session-id', 's', '--confidence', '.9')
        self.assertEqual(cli('list', '--user-id', 'u')[0]['id'], new['id'])
        self.assertEqual(len(cli('list', '--user-id', 'u', '--include-history')), 2)
        self.assertEqual(len(cli('history', old['id'], '--user-id', 'u')), 2)
        selected = cli('resolve', '--keep', old['id'], '--user-id', 'u')
        self.assertEqual(selected['memory_status'], 'active')
        self.assertEqual(cli('reconcile', '--user-id', 'u')['changed_records'], 0)
