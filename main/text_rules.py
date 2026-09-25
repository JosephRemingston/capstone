"""Shared, deliberately narrow signals for conversational rules."""
import re


def matches(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.I) is not None


def preference_constraint(text: str) -> bool:
    return matches(
        r"\b(?:never|always)\s+(?:include|use|serve|give|add)\b|"
        r"\b(?:do not|don't|avoid)\s+(?:include|including|use|using|serve|serving|add|adding)\b|"
        r"\b(?:i am|i'm)\s+allergic\s+to\b|\bno\s+\w+\s+in\s+my\b", text)


def event_update(text: str) -> bool:
    # Do not treat negated completion or cancellation as a finished event.
    if matches(r"\b(?:not|never|haven't|hasn't|isn't|wasn't|didn't)\b", text):
        return False
    return matches(
        r"\b(?:i|we)\s+(?:(?:have|had|just|already)\s+)*"
        r"(?:submitted|completed|finished|cancelled|canceled|booked|scheduled)\b|"
        r"\b(?:meeting|appointment|event|task|deadline|report|booking)\b.{0,60}"
        r"\b(?:is|was|has been|have been)\s+(?:already\s+)?"
        r"(?:cancelled|canceled|completed|finished|submitted|rescheduled|postponed)\b", text)


def recurring(text: str) -> bool:
    return matches(r"\b(?:daily|weekly|monthly|yearly|every\s+(?:day|week|month|year|morning|evening|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b", text)


def deadline(text: str) -> bool:
    return matches(r"\b(?:deadline|due|today|tonight|tomorrow|next\s+(?:week|month)|"
                   r"by\s+(?:\w+)|in\s+\d+\s+(?:minutes?|hours?|days?|weeks?)|"
                   r"\d{4}-\d{2}-\d{2})\b", text)
