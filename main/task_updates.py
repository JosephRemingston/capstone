"""Apply a recognized action to one already-resolved task reference."""
from dataclasses import replace
from datetime import timedelta, datetime
import re

from .deadlines import parse_deadline
from .lifecycle import LifecycleManager
from .models import MemoryRecord, MemoryTier
from .recurrence import advance, parse_recurrence, scheduled_at
from .tasks import TaskAction
from .text_rules import recurring


def _occurrence_matches(value: str, due: datetime, scheduled: datetime) -> bool:
    if not isinstance(value, str):
        raise ValueError('occurrence_at must be an ISO date or datetime')
    parsed = parse_deadline('', scheduled, value)
    if len(value) == 10:
        return parsed.date() in {scheduled.date(), due.astimezone(scheduled.tzinfo).date()}
    return parsed in {due, scheduled}


def revise_task(target: MemoryRecord, event: MemoryRecord, action: TaskAction) -> MemoryRecord | None:
    scope = event.source_metadata.get('task_scope') or action.scope or 'occurrence'
    explicit_occurrence = event.source_metadata.get('occurrence_at')
    is_series = recurring(target.content)
    if explicit_occurrence is not None and (not is_series or scope == 'series'):
        raise ValueError('occurrence_at requires a recurring task and occurrence scope')
    target = replace(target, source_metadata=dict(target.source_metadata))
    if is_series and target.recurrence is None:
        # Upgrade older JSONL tasks lazily using their original creation timestamp.
        target.recurrence = parse_recurrence(target.content, target.created_at,
                                             target.source_metadata.get('due_at'))
        target.next_due_at = scheduled_at(target.recurrence) if target.recurrence else None
    if is_series and scope != 'series':
        if not target.recurrence or not target.next_due_at:
            event.source_metadata['task_resolution'] = 'unsupported_schedule'
            return None
        scheduled = scheduled_at(target.recurrence)
        due = target.next_due_at.astimezone(scheduled.tzinfo)
        selector = explicit_occurrence
        if selector is None:
            named_date = re.search(r'\b\d{4}-\d{2}-\d{2}\b', action.reference)
            relative_day = re.search(r'\b(today|tomorrow|yesterday)\b', action.reference)
            if named_date:
                selector = named_date[0]
            elif relative_day:
                offset = {'today': 0, 'tomorrow': 1, 'yesterday': -1}[relative_day[1]]
                selector = (event.created_at.astimezone(scheduled.tzinfo) + timedelta(days=offset)).date().isoformat()
        if selector is not None:
            if not _occurrence_matches(selector, due, scheduled):
                previous = any(_occurrence_matches(selector, datetime.fromisoformat(item['due_at']),
                                                    datetime.fromisoformat(item['scheduled_at']))
                               for item in target.task_occurrences)
                event.source_metadata['task_resolution'] = 'already_resolved' if previous else 'occurrence_not_current'
                return None
        elif action.kind == 'completed' and due.date() != event.created_at.astimezone(scheduled.tzinfo).date():
            event.source_metadata['task_resolution'] = 'occurrence_required'
            return None

    if action.kind == 'rescheduled':
        previous_due = target.next_due_at if is_series else target.due_at
        date_text, reference = action.deadline_text, event.created_at
        if date_text.startswith('delay ') and event.source_metadata.get('due_at') is None:
            if previous_due is None:
                event.source_metadata['task_resolution'] = 'missing_deadline'
                return None
            date_text, reference = 'in ' + date_text[6:], previous_due
        new_due = parse_deadline(date_text, reference, event.source_metadata.get('due_at'))
        if new_due is None:
            event.source_metadata['task_resolution'] = 'missing_deadline'
            return None
        if is_series:
            if not target.recurrence:
                event.source_metadata['task_resolution'] = 'unsupported_schedule'
                return None
            if scope == 'series':
                if any(new_due <= max(datetime.fromisoformat(item['due_at']),
                                      datetime.fromisoformat(item['scheduled_at']))
                       for item in target.task_occurrences):
                    event.source_metadata['task_resolution'] = 'series_overlaps_history'
                    return None
                target.recurrence = {**target.recurrence, 'anchor': new_due.isoformat(), 'index': 0}
            else:
                # Moving an occurrence past its successor would reorder the cursor.
                _, successor = advance(target.recurrence)
                if new_due >= successor:
                    event.source_metadata['task_resolution'] = 'occurrence_overlaps_next'
                    return None
            target.next_due_at = new_due
        else:
            target.due_at = new_due
            target.expires_at = LifecycleManager().expiry_for(target)
        event.source_metadata['task_change'] = {
            'previous_due_at': previous_due.isoformat() if previous_due else None,
            'due_at': new_due.isoformat(), 'scope': scope if is_series else 'task'}
    elif is_series and scope != 'series':
        scheduled = scheduled_at(target.recurrence)
        occurrence = {'scheduled_at': scheduled.isoformat(), 'due_at': target.next_due_at.isoformat(),
                      'status': action.kind, 'recorded_at': event.created_at.isoformat(), 'event_id': event.id}
        target.task_occurrences = [*target.task_occurrences, occurrence]
        target.recurrence, target.next_due_at = advance(target.recurrence)
        event.source_metadata['task_change'] = occurrence
    else:
        target.task_status = action.kind
        target.tier = MemoryTier.ARCHIVE
        target.expires_at = target.archive_after = target.next_due_at = None
        target.source_metadata['resolved_by'] = event.id
    target.updated_at = event.created_at
    target.source_metadata['last_task_update'] = event.id
    event.related_task_id = target.id
    event.source_metadata['task_resolution'] = 'applied'
    event.source_metadata['task_action'] = action.kind
    return target
