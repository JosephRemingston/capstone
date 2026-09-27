"""Shared, deliberately narrow signals for conversational rules."""
import re


def normalize(text: str) -> str:
    return text.replace("’", "'").replace('“', '"').replace('”', '"').lower().strip()


def uncertain(text: str, *, quoted_objects: bool = False) -> bool:
    """Reported, conditional and interrogative content is not a user assertion."""
    text = normalize(text)
    if quoted_objects and not text.startswith(('"', "'")):
        text = re.sub(r'"[^"\n]*"', '', text)
    return ('?' in text or '"' in text or
            matches(r"\b(?:if|unless|whether|maybe|perhaps|might|could|suppose|pretend)\b|"
                    r"\b(?:not sure|not certain|don't know|do not know|said|says|asked)\b|"
                    r"^'", text))


def matches(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.I) is not None


def preference_constraint(text: str) -> bool:
    text = normalize(text)
    if uncertain(text, quoted_objects=True):
        return False
    return matches(
        r"\b(?:never|always)\s+(?:include|use|serve|give|add)\b|"
        r"\b(?:do not|don't|avoid)\s+(?:include|including|use|using|serve|serving|add|adding)\b|"
        r"\b(?:i am|i'm)\s+allergic\s+to\b|\bno\s+\w+\s+in\s+my\b|"
        r"\b(?:i|my\s+\w+)\s+(?:cannot|can't|must not)\s+(?:eat|have|tolerate|consume)\b|"
        r"\bi\s+(?:would rather|would prefer|tend to prefer|usually (?:take|choose|drink|eat)|"
        r"(?:don't|do not)\s+(?:really\s+)?(?:like|dislike|hate|prefer|enjoy|mind)|wouldn't mind|can't stand)\b|"
        r"\bi(?:'d rather|'m (?:not )?(?:a fan of|partial to)| am (?:not )?(?:a fan of|partial to))\b|"
        r"\b(?:works?|suits?)\s+(?:best|better)\s+for\s+me\b|"
        r"\bmy\s+\w+\s+of choice\b|\b(?:isn't|is not)\s+my\s+(?:thing|cup of tea)\b", text)


def event_update(text: str) -> bool:
    text = text.replace("’", "'")
    # Do not treat negated completion or cancellation as a finished event.
    if matches(r"\b(?:not|never|haven't|hasn't|hadn't|isn't|wasn't|weren't|didn't)\b", text):
        return False
    return matches(
        r"\b(?:got|was|is|been)\s+(?:called off|pushed back|wrapped up)\b|"
        r"\b(?:i|we)\s+(?:(?:have|had|just|already)\s+)*"
        r"(?:submitted|completed|finished|cancelled|canceled|booked|scheduled|paid|sent|bought)\b|"
        r"\b(?:meeting|appointment|event|task|deadline|report|booking)\b.{0,60}"
        r"\b(?:is|was|has been|have been)\s+(?:already\s+)?"
        r"(?:cancelled|canceled|completed|finished|submitted|rescheduled|postponed)\b", text)


def negated_task(text: str) -> bool:
    return matches(r"\b(?:not|never|haven't|hasn't|hadn't|didn't)\s+(?:(?:yet|quite|actually|fully)\s+)*"
                   r"(?:complete[ds]?|finish(?:ed)?|submit(?:ted)?|paid|pay|sent|send|bought|buy)\b", text)


def recurring(text: str) -> bool:
    return matches(r"\b(?:daily|weekly|monthly|yearly|every\s+(?:(?:\d+|other)\s+)?(?:days?|weeks?|months?|years?|morning|evening|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b", text)


def deadline(text: str) -> bool:
    return matches(r"\b(?:deadline|due|today|tonight|tomorrow|next\s+(?:week|month)|"
                   r"by\s+(?:\w+)|in\s+\d+\s+(?:minutes?|hours?|days?|weeks?)|"
                   r"\d{4}-\d{2}-\d{2})\b", text)
