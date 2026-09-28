"""Conservative English deadline parsing relative to the input's timezone.

Supported: ISO dates/datetimes, today/tonight/tomorrow, next week,
weekdays, numeric relative durations, and optional 24h or am/pm times.
Unknown wording returns None rather than inventing a deadline.
"""
from datetime import datetime, time, timedelta
import re


def parse_deadline(text: str, reference: datetime, explicit: str | None = None) -> datetime | None:
    if reference.tzinfo is None:
        raise ValueError("Deadline reference must include a timezone")
    if explicit is not None:
        if not isinstance(explicit, str):
            raise ValueError("metadata.due_at must be an ISO date or datetime string")
        return _iso(explicit, reference)
    text = text.casefold().replace("’", "'")
    # Do not infer a positive deadline from a negated date or choose among alternatives.
    if re.search(r"\b(?:not|never|no longer)\b|\bor\b", text):
        return None
    relative = re.search(r"\bin\s+(\d+)\s+(minutes?|hours?|days?|weeks?)\b", text)
    if relative:
        unit = relative[2].rstrip('s') + 's'
        try:
            return reference + timedelta(**{unit: int(relative[1])})
        except OverflowError as exc:
            raise ValueError("Deadline is out of range") from exc
    date_match = re.search(r"\b\d{4}-\d{2}-\d{2}(?:t\d{2}:\d{2}(?::\d{2})?(?:z|[+-]\d{2}:\d{2})?)?\b", text)
    if date_match:
        result = _iso(date_match[0], reference)
        if 't' in date_match[0]:
            return result
        day = result.date()
    elif re.search(r"\btomorrow\b", text):
        day = (reference + timedelta(days=1)).date()
    elif re.search(r"\b(?:today|tonight)\b", text):
        day = reference.date()
    elif re.search(r"\bnext week\b", text):
        day = (reference + timedelta(days=7)).date()
    else:
        weekdays = ('monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday')
        match = re.search(r"\b(?:(next|this)\s+)?(" + '|'.join(weekdays) + r")\b", text)
        if not match:
            return None
        offset = (weekdays.index(match[2]) - reference.weekday()) % 7
        # 'next Monday' means the next occurrence, strictly after today.
        if match[1] == 'next' and offset == 0:
            offset = 7
        day = (reference + timedelta(days=offset)).date()
    clock = re.search(r"\b(?:at|by)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text)
    hour, minute, second = 23, 59, 59
    if clock:
        hour, minute = int(clock[1]), int(clock[2] or 0)
        if clock[3]:
            if not 1 <= hour <= 12:
                raise ValueError("Invalid deadline hour")
            hour = hour % 12 + (12 if clock[3] == 'pm' else 0)
        second = 0
    return datetime.combine(day, time(hour, minute, second), tzinfo=reference.tzinfo)


def _iso(value: str, reference: datetime) -> datetime:
    try:
        if len(value) == 10:
            return datetime.combine(datetime.fromisoformat(value).date(), time(23, 59, 59), tzinfo=reference.tzinfo)
        result = datetime.fromisoformat(value.replace('z', '+00:00').replace('Z', '+00:00'))
        return result if result.tzinfo else result.replace(tzinfo=reference.tzinfo)
    except ValueError as exc:
        raise ValueError(f"Invalid deadline: {value}") from exc
