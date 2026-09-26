"""Task actions and conservative object/reference matching."""
from dataclasses import dataclass
import re
from typing import Iterable

from .models import MemoryRecord
from .text_rules import matches, normalize, uncertain


@dataclass(frozen=True)
class TaskAction:
    kind: str
    reference: str
    deadline_text: str = ''
    scope: str | None = None


def task_action(text: str) -> TaskAction | None:
    text = normalize(text).rstrip('.!')
    if uncertain(text):
        return None
    # A tightly scoped date correction can contain negation without negating the update.
    correction = re.fullmatch(r'(?:the |my |our )?(.+?) is (?:not|no longer) due (.+?)'
                              r',? (?:but|instead) (?:is )?(?:now )?(?:due )?(.+)', text)
    if correction and not matches(r"\b(?:not|never|no|and|or|but)\b", correction[3]):
        return TaskAction('rescheduled', correction[1], correction[3])
    # A mixed message must be segmented before it can change stored tasks.
    if matches(
        r"\b(?:not|never|haven't|hasn't|hadn't|isn't|wasn't|weren't|didn't|don't|"
        r"cannot|can't|would|will|plan to|want to|need to|and|but|nor)\b|[;\n]|[.!]\s+\w", text
    ):
        return None

    scope = 'series' if matches(r'\b(?:entire series|whole series|all occurrences|permanently)\b', text) else None
    text = re.sub(r'\b(?:entire series|whole series|all occurrences|permanently)\b', '', text).strip()
    # Capture the replacement date separately: the old date must never win.
    change = re.match(
        r"^(?:(?:please\s+)?(?:reschedule|postpone|move|push back|change the deadline for)|"
        r"(?:i|we)(?: have|'ve)?\s+(?:rescheduled|postponed|moved|pushed back))\s+(.+)$", text)
    passive = re.match(r"^(?:the |my |our )?(.+?)\s+(?:is|was|has been|got)\s+"
                       r"(?:rescheduled|postponed|moved|pushed back|now due)\s*(.*)$", text)
    changed_deadline = re.match(r"^(?:the )?deadline (?:for|of) (.+?)\s+"
                                r"(?:is now|has changed|changed|has moved|moved)\s+(.+)$", text)
    if change or passive or changed_deadline:
        if change:
            parts = re.split(r'\s+(?:to|until|by(?=\s+\d+\s+(?:minutes?|hours?|days?|weeks?)\b))\s+', change[1])
            reference = re.split(r'\s+from\s+', parts[0])[0]
            date = parts[-1] if len(parts) > 1 else ''
            if re.search(r'\s+by\s+\d+\s+(?:minutes?|hours?|days?|weeks?)\b', change[1]):
                date = 'delay ' + date
        else:
            match = passive or changed_deadline
            reference, date = match[1], match[2]
            date = re.split(r'\s+(?:to|until)\s+', date)[-1]
            date = re.sub(r'^(?:to|until|for)\s+', '', date)
        return TaskAction('rescheduled', reference, date, scope)

    active = re.match(r"^(?:i|we)(?:'ve| have| had)?\s+(?:(?:just|already|finally)\s+)*"
                      r"(submitted|completed|finished|paid|sent|bought|wrapped up|taken care of|"
                      r"cancelled|canceled|called off)\s+(.+)$", text)
    done = re.match(r"^(?:i am|i'm|we are|we're)\s+done with\s+(.+)$", text)
    passive = re.match(r"^(?:the |my |our )?(.+?)\s+(?:is|was|has been|got)\s+"
                       r"(?:already\s+)?(completed|finished|submitted|done|cancelled|canceled|called off)$", text)
    cancel = re.match(r'^(?:please\s+)?(?:cancel|skip)\s+(.+)$', text)
    stop = re.match(r'^stop reminding me to\s+(.+)$', text)
    if active:
        kind = 'cancelled' if active[1] in {'cancelled', 'canceled', 'called off'} else 'completed'
        return TaskAction(kind, active[2], scope=scope)
    if done:
        return TaskAction('completed', done[1], scope=scope)
    if passive:
        kind = 'cancelled' if passive[2] in {'cancelled', 'canceled', 'called off'} else 'completed'
        return TaskAction(kind, passive[1], scope=scope)
    if cancel or stop:
        return TaskAction('cancelled', (cancel or stop)[1], scope='series' if stop else scope)
    return None


def task_outcome(text: str) -> str | None:
    action = task_action(text)
    return action.kind if action and action.kind != 'rescheduled' else None


_STOP = set('i we my our the a an to me please remind need have has had been is was were just already '
            'submit submitted complete completed finish finished cancel cancelled canceled buy bought '
            'pay paid send sent book booked schedule scheduled task today tomorrow tonight yesterday '
            'by at on in this that next week day days hours minutes am pm every other daily weekly '
            'monthly yearly morning evening monday tuesday wednesday thursday friday saturday sunday '
            'occurrence time for of'.split())


def task_key(text: str) -> frozenset[str]:
    text = normalize(text)
    text = re.sub(r"\b(?:today|tomorrow|yesterday)'s\b", '', text)
    text = re.sub(r'\d{4}-\d{2}-\d{2}(?:t\S+)?', '', text)
    text = re.sub(r'\b(?:at|by)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b|'
                  r'\b(?:in|every)\s+\d+\s+(?:minutes?|hours?|days?|weeks?|months?|years?)\b', '', text)
    return frozenset(word for word in re.findall(r"[a-z0-9]+", text) if word not in _STOP)


def match_tasks(candidates: Iterable[MemoryRecord], action: TaskAction, session_id: str) -> list[MemoryRecord]:
    """Exact objects, then unique noun subsets. Bare pronouns stay in-session."""
    candidates = list(candidates)
    key = task_key(action.reference)
    if not key or key <= {'it', 'that', 'this', 'one'}:
        return [item for item in candidates if item.session_id == session_id]
    exact = [item for item in candidates if task_key(item.content) == key]
    if exact:
        return exact
    # Never drop or change a numbered identifier, e.g. report 1 versus report 2.
    numbers = {word for word in key if word.isdigit()}
    return [item for item in candidates if key <= task_key(item.content)
            and {word for word in task_key(item.content) if word.isdigit()} == numbers]
