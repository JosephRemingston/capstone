"""Conservative splitting at explicit independent clauses, not arbitrary 'and'."""
import re
from dataclasses import replace

from .models import MemoryInput


def split_input(incoming: MemoryInput) -> list[MemoryInput]:
    text = incoming.content
    # Quoted text and procedural sequences need richer parsing; leave them whole.
    if '"' in text or re.search(r'\b(?:first|then|finally|how to|steps)\b', text, re.I):
        return [incoming]
    parts = re.split(
        r'[;\n]+|(?<=[.!?])\s+(?=(?:I\b|My\b|We\b|Our\b|Remind\b|Please\b|Never\b|The\b))|'
        r',?\s+(?:and|but)\s+(?=(?:I\b|my\b|we\b|our\b|remind me\b|please\b|never\b|the\b))',
        text, flags=re.I)
    parts = [part.strip(' ,') for part in parts if part.strip(' ,')]
    if len(parts) <= 1:
        return [incoming]
    # One explicit target/deadline cannot safely be assigned to several clauses.
    if any(key in incoming.metadata for key in ('task_id', 'due_at')):
        raise ValueError('Split messages with task_id/due_at into separate inputs first')
    return [replace(incoming, content=part, metadata={**incoming.metadata,
            'source_content': text, 'segment_index': index, 'segment_count': len(parts)})
            for index, part in enumerate(parts)]
