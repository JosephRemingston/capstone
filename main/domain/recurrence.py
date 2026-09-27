"""Calendar-based recurrence with a stable anchor and explicit occurrence history.

The cursor points to the oldest unresolved occurrence. Reads never advance it.
Calendar months clamp to their last day without drifting the original anchor day.
"""
from calendar import monthrange
from datetime import datetime, timedelta
import re

from .deadlines import parse_deadline
from .text_rules import normalize, recurring

_WEEKDAYS = 'monday tuesday wednesday thursday friday saturday sunday'.split()


def scheduled_at(rule: dict, index: int | None = None) -> datetime:
    anchor = datetime.fromisoformat(rule['anchor'])
    count = rule['interval'] * (rule['index'] if index is None else index)
    unit = rule['unit']
    try:
        if unit in {'day', 'week'}:
            return anchor + timedelta(days=count * (7 if unit == 'week' else 1))
        months = count * (12 if unit == 'year' else 1)
        year, month = divmod(anchor.year * 12 + anchor.month - 1 + months, 12)
        month += 1
        return anchor.replace(year=year, month=month,
                              day=min(anchor.day, monthrange(year, month)[1]))
    except (ValueError, OverflowError) as exc:
        raise ValueError('Recurrence is out of range') from exc


def parse_recurrence(text: str, reference: datetime, explicit_start: str | None = None) -> dict | None:
    text = normalize(text)
    if not recurring(text):
        return None
    match = re.search(r'\bevery\s+(?:(\d+|other)\s+)?(days?|weeks?|months?|years?)\b', text)
    weekday = re.search(r'\bevery\s+(' + '|'.join(_WEEKDAYS) + r')\b', text)
    cadence = re.search(r'\b(daily|weekly|monthly|yearly|every morning|every evening)\b', text)
    if match:
        interval = 2 if match[1] == 'other' else int(match[1] or 1)
        unit = match[2].rstrip('s')
    elif weekday:
        unit, interval = 'week', 1
    elif cadence:
        unit = {'daily': 'day', 'weekly': 'week', 'monthly': 'month', 'yearly': 'year',
                'every morning': 'day', 'every evening': 'day'}[cadence[1]]
        interval = 1
    else:
        return None
    if interval < 1 or interval > 1000:
        raise ValueError('Recurrence interval must be between 1 and 1000')
    # Multiple schedules need an explicit policy; do not silently take the first.
    schedule_text = text[(match or weekday or cadence).start():]
    if re.search(r'\b(?:and|or|except|until)\b', schedule_text) or len(re.findall(r'\bevery\b', text)) > 1:
        return None
    if explicit_start:
        anchor = parse_deadline('', reference, explicit_start)
    else:
        clock = re.search(r'\b(?:at|by)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b', text)
        start = re.search(r'\b(?:starting|beginning|from)\s+(.+)$', text)
        day_text = start[1] if start else (weekday[1] if weekday else 'today')
        clock_text = clock[0] if clock else ''
        anchor = parse_deadline(day_text + ' ' + clock_text, reference)
        if anchor is None:
            return None
    rule = {'unit': unit, 'interval': interval, 'anchor': anchor.isoformat(), 'index': 0}
    if not explicit_start and anchor < reference:
        anchor = scheduled_at(rule, 1)
        rule['anchor'] = anchor.isoformat()
    return rule


def advance(rule: dict) -> tuple[dict, datetime]:
    result = {**rule, 'index': rule['index'] + 1}
    return result, scheduled_at(result)
