"""Split independent clauses without losing quoted or conditional scope."""
import re
from dataclasses import replace
from datetime import timedelta

from .models import MemoryInput

_START = r"(?:i\b|my\b|we\b|our\b|remind me\b|please\b|never\b|the\b|" \
         r"reschedule\b|postpone\b|buy\b|submit\b|send\b|pay\b|cancel\b|skip\b)"
_BOUNDARY = re.compile(
    r'[;\n]+|(?<=[.!?])\s+(?=' + _START + r')|'
    r',?\s+(?:and|but|also)\s+(?=' + _START + r')', re.I)


def split_input(incoming: MemoryInput) -> list[MemoryInput]:
    text = incoming.content
    # Scope cannot safely be inherited across an arbitrary conditional or sequence.
    if re.search(r'\b(?:if|unless|whether)\b|^\s*(?:first|how to|steps)\b', text, re.I):
        return [incoming]
    # Mask quoted spans while keeping character positions intact.
    masked = re.sub(r'"[^"\n]*"|“[^”\n]*”|(?<!\w)\x27[^\x27\n]+\x27(?!\w)',
                    lambda match: 'x' * len(match[0]), text)
    parts, start = [], 0
    for boundary in _BOUNDARY.finditer(masked):
        parts.append(text[start:boundary.start()].strip(' ,'))
        start = boundary.end()
    parts.append(text[start:].strip(' ,'))
    parts = [part for part in parts if part]
    if len(parts) <= 1:
        return [incoming]
    if any(key in incoming.metadata for key in ('task_id', 'due_at', 'occurrence_at', 'task_scope')):
        raise ValueError('Split messages with task_id/due_at/occurrence_at/task_scope into separate inputs first')
    return [replace(incoming, content=part, timestamp=incoming.timestamp + timedelta(microseconds=index), metadata={**incoming.metadata,
            'source_content': text, 'segment_index': index, 'segment_count': len(parts)})
            for index, part in enumerate(parts)]
