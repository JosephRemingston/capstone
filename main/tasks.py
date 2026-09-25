"""Conservative task state recognition and matching; no fuzzy auto-resolution."""
import re

from .text_rules import event_update, matches


def task_outcome(text: str) -> str | None:
    text = text.replace('’', "'").lower().strip()
    # Questions, intentions, quotes and hypothetical statements are not evidence.
    if '?' in text or '"' in text or matches(r"\b(?:if|unless|would|could|might|will|plan to|want to|remind me|need to|please|and|but)\b", text):
        return None
    if not matches(r"^(?:(?:i|we)\s+(?:(?:have|had|just|already)\s+)*(?:submitted|completed|finished|cancelled|canceled|paid|sent|bought)\b|"
                   r"(?:the|my|our)\s+(?:meeting|appointment|event|task|deadline|report|booking)\b.{0,60}\b(?:is|was|has been|have been)\b)", text):
        return None
    if not event_update(text):
        return None
    if matches(r"\b(?:cancelled|canceled)\b", text):
        return 'cancelled'
    if matches(r"\b(?:submitted|completed|finished|paid|sent|bought)\b", text):
        return 'completed'
    return None


# These words describe action/state/time rather than the task's object.
_STOP = set('i we my our the a an to me please remind need have has had been is was were just already '
            'submit submitted complete completed finish finished cancel cancelled canceled buy bought '
            'pay paid send sent book booked schedule scheduled task today tomorrow tonight yesterday '
            'by at on in this next week day days hours minutes am pm'.split())


def task_key(text: str) -> frozenset[str]:
    text = re.sub(r'\d{4}-\d{2}-\d{2}', '', text.lower())
    text = re.sub(r'\b(?:at|by)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b|\bin\s+\d+\s+(?:minutes?|hours?|days?|weeks?)\b', '', text)
    return frozenset(word for word in re.findall(r"[a-z0-9]+", text) if word not in _STOP)
