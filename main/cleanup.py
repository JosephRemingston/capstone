"""Conservative physical expiry and lossless compaction of retained history."""
from datetime import timedelta
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile

from .locking import store_lock
from .models import utc_now


def atomic_write(path, payload):
    """Stage alongside target, fsync data, replace atomically, fsync directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_index_registry(path):
    paths = json.loads(path.read_text()) if path.exists() else []
    if not isinstance(paths, list) or any(not isinstance(value, str) or not Path(value).is_absolute() for value in paths):
        raise ValueError('Invalid index registry; cleanup aborted')
    return paths


def register_index(store_path, index_path):
    """Remember custom indexes so cleanup cannot silently leave them behind."""
    with store_lock(store_path):
        registry = Path(str(Path(store_path).resolve()) + '.indexes.json')
        paths = read_index_registry(registry)
        index = str(Path(index_path).resolve())
        if index not in paths:
            atomic_write(registry, (json.dumps(sorted([*paths, index])) + '\n').encode())


def references(record):
    result = set(record.evidence_ids + record.conflict_ids)
    for value in (record.related_task_id, record.superseded_by, record.consolidated_into,
                  record.source_metadata.get('task_id')):
        if isinstance(value, str):
            result.add(value)
    decision = record.conflict_resolution or {}
    for key in ('winner_id', 'observation_id'):
        if isinstance(decision.get(key), str):
            result.add(decision[key])
    result.update(value for value in record.source_metadata.get('candidate_task_ids', []) if isinstance(value, str))
    for occurrence in record.task_occurrences:
        for key, value in occurrence.items():
            if key.endswith('_id') and isinstance(value, str):
                result.add(value)
    result.discard(record.id)
    return result


def plan(store, *, user_id, now=None, grace_days=7):
    if not user_id or not math.isfinite(grace_days) or grace_days < 0:
        raise ValueError('Cleanup requires user_id and a finite nonnegative grace period')
    now = now or utc_now()
    if now.tzinfo is None:
        raise ValueError('Cleanup time must include a timezone')
    try:
        cutoff = now - timedelta(days=grace_days)
    except OverflowError as exc:
        raise ValueError('Cleanup grace period is out of range') from exc
    revisions = store._read_log()  # Fail closed on corrupt input.
    if store.path.exists():
        raw_rows = [json.loads(line) for line in store.path.read_text().splitlines() if line.strip()]
        if any(set(raw) - set(record.to_dict()) for raw, record in zip(raw_rows, revisions)):
            raise ValueError('Unknown record fields require a migration before cleanup')
    latest = {record.id: record for record in revisions}
    held = {record.id for record in revisions if record.source_metadata.get('legal_hold') or record.source_metadata.get('retain')}
    eligible = {key for key, record in latest.items() if record.user_id == user_id
                and record.expires_at is not None and record.expires_at <= cutoff
                and key not in held
                and not (record.category.value == 'task' and record.task_status not in {'completed', 'cancelled'})}
    # Follow references from ALL revisions of every retained root, across users.
    # A candidate cycle can be removed as a unit only if nothing retained needs it.
    dependencies = {}
    for record in revisions:
        dependencies.setdefault(record.id, set()).update(references(record))
    retained = set(latest) - eligible
    pending = list(retained)
    while pending:
        for target in dependencies.get(pending.pop(), ()):
            if target in latest and target not in retained:
                retained.add(target)
                pending.append(target)
    removed = eligible - retained
    compacted, previous, duplicate_count = [], {}, 0
    for record in revisions:
        if record.id in removed:
            continue
        payload = record.to_dict()
        # Only exact repeated snapshots of this ID are redundant. A -> B -> A
        # is NOT deduplicated: the return to A is a meaningful transition.
        if record.user_id == user_id and previous.get(record.id) == payload:
            duplicate_count += 1
            continue
        previous[record.id] = payload
        compacted.append(record)
    payload = ''.join(json.dumps(r.to_dict(), sort_keys=True, separators=(',', ':')) + '\n' for r in compacted).encode()
    original = store.path.read_bytes() if store.path.exists() else b''
    result = {'user_id': user_id, 'cutoff': cutoff.isoformat(), 'grace_days': grace_days,
              'removed_ids': sorted(removed), 'removed_memories': len(removed),
              'protected_referenced_expired': len(eligible & retained),
              'duplicate_revisions_removed': duplicate_count,
              'revisions_before': len(revisions), 'revisions_after': len(compacted),
              'bytes_before': len(original), 'bytes_after': len(payload),
              'bytes_reclaimed': len(original) - len(payload),
              'source_sha256': sha256(original).hexdigest()}
    return result, payload


def purge_indexes(store, user_id):
    default = store.path.with_suffix('.index.sqlite3').resolve()
    registry = Path(str(store.path.resolve()) + '.indexes.json')
    paths = {str(default)}
    if registry.exists():
        paths.update(read_index_registry(registry))
    purged = []
    for name in sorted(paths):
        path = Path(name)
        if not path.exists():
            continue
        # Do not unlink an open database: SQLite coordinates its readers/writers.
        db = sqlite3.connect(path, timeout=30)
        try:
            db.execute('PRAGMA secure_delete=ON')
            with db:
                for table in ('vectors', 'edges', 'aliases', 'nodes', 'index_state'):
                    db.execute(f'DELETE FROM {table} WHERE user_id=?', (user_id,))
            if db.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0]:
                raise ValueError('Index is busy; cleanup can be retried without losing source history')
            db.execute('VACUUM')
            if db.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0]:
                raise ValueError('Index checkpoint is busy; retry cleanup')
            purged.append(str(path))
        finally:
            db.close()
    return purged


def cleanup(store, *, user_id, now=None, grace_days=7, apply=False, scheduled=False, interval_hours=24):
    if not math.isfinite(interval_hours) or not math.isfinite(interval_hours * 3600) or interval_hours <= 0:
        raise ValueError('Schedule interval must be finite and positive')
    now = now or utc_now()
    if now.tzinfo is None:
        raise ValueError('Cleanup time must include a timezone')
    with store_lock(store.path):
        state_path = Path(str(store.path.resolve()) + '.cleanup-schedule.json')
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if not isinstance(state, dict):
            raise ValueError('Invalid cleanup schedule state')
        # Separate schedules when policy changes; state contains no memory text.
        key = sha256(json.dumps([user_id, float(grace_days), float(interval_hours)]).encode()).hexdigest()
        last = state.get(key)
        if last is not None and (not isinstance(last, dict)
                or type(last.get('completed_at')) not in (int, float)
                or not math.isfinite(last['completed_at'])):
            raise ValueError('Invalid cleanup schedule timestamp')
        if scheduled and last and now.timestamp() < last['completed_at'] + interval_hours * 3600:
            return {'applied': False, 'status': 'not_due', 'next_run_timestamp': last['completed_at'] + interval_hours * 3600}
        result, payload = plan(store, user_id=user_id, now=now, grace_days=grace_days)
        result.update(applied=False, status='preview', indexes_invalidated=[])
        if not apply:
            return result
        # Clear derived copies BEFORE replacing the log. A failure leaves source
        # history intact; empty caches are safely rebuilt. No raw-data backup is kept.
        if result['removed_memories']:
            result['indexes_invalidated'] = purge_indexes(store, user_id)
        original = store.path.read_bytes() if store.path.exists() else b''
        if sha256(original).hexdigest() != result['source_sha256']:
            raise ValueError('Source changed outside the store lock; cleanup aborted')
        if payload != original:
            atomic_write(store.path, payload)
        result.update(applied=True, status='completed')
        if scheduled:
            state[key] = {'completed_at': now.timestamp(), 'user_id': user_id}
            atomic_write(state_path, (json.dumps(state, sort_keys=True) + '\n').encode())
        return result
