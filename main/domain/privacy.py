"""Deterministic redaction for secrets and high-risk identifiers."""
from __future__ import annotations

import re

_PATTERNS = (
    ('api_key', re.compile(r'(?i)\b(api[_ -]?key|access[_ -]?token|secret)\s*[:=]\s*["\']?([A-Za-z0-9_\-]{12,})["\']?')),
    ('password', re.compile(r'(?i)\b(password|passcode|pin)\s*[:=]\s*([^\s,;]{4,})')),
    ('email', re.compile(r'(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?!\w)')),
    ('phone', re.compile(r'(?i)(?:\b(?:phone|mobile|tel)\s*[:=]\s*|(?<!\w)\+)(?:\d[\d ()-]{8,}\d)(?!\w)')),
    ('payment_card', re.compile(r'(?<!\d)(?:\d[ -]?){13,19}(?!\d)')),
    ('government_id', re.compile(r'(?i)\b(?:ssn|aadhaar|aadhar)\s*[:#-]?\s*[A-Z0-9 -]{8,18}\b')),
)


def redact_sensitive(text: str) -> tuple[str, list[str]]:
    """Return sanitized text and detected classes without retaining raw matches."""
    redacted, found = text, []
    for label, pattern in _PATTERNS:
        if pattern.search(redacted):
            found.append(label)
            if label in {'api_key', 'password'}:
                redacted = pattern.sub(lambda m: f'{m.group(1)}: [REDACTED_{label.upper()}]', redacted)
            else:
                redacted = pattern.sub(f'[REDACTED_{label.upper()}]', redacted)
    return redacted, found


def redact_metadata(value):
    """Recursively sanitize caller metadata and return it with detected classes."""
    found = []
    if isinstance(value, str):
        cleaned, found = redact_sensitive(value)
        return cleaned, found
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            sanitized, kinds = redact_metadata(item)
            cleaned[key] = sanitized
            found.extend(kinds)
        return cleaned, list(dict.fromkeys(found))
    if isinstance(value, list):
        cleaned = []
        for item in value:
            sanitized, kinds = redact_metadata(item)
            cleaned.append(sanitized)
            found.extend(kinds)
        return cleaned, list(dict.fromkeys(found))
    return value, found
