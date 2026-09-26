"""Narrow, auditable assertions for conflict detection and consolidation.

Only complete, single assertions are extracted. Similar words alone are not
proof of a contradiction, and liking two different things is not a conflict.
"""
import re
import unicodedata

from .text_rules import normalize, uncertain


def normalized_value(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', normalize(text)).split()).strip(' .!')


def assertion_text(text: str) -> str | None:
    text = text.strip().replace('’', "'")
    text = re.sub(r"^I(?: would rather|'d rather) (?:receive|have|get) ", 'I prefer ', text, flags=re.I)
    if uncertain(text) or re.search(
        r'\b(?:and|but|or|nor|used to|previously|formerly|yesterday|tomorrow|today|'
        r'sometimes|usually|possibly|probably|think|believe|wish|hope|would|might|'
        r'while|when|during|before|after|except|only|also)\b|[;\n]|[.!]\s+\w', text, re.I
    ):
        return None
    return re.sub(r'^(?:actually,?\s+|currently,?\s+|please remember that\s+)', '', text.strip().replace('’', "'"), flags=re.I).rstrip('.! ')


def extract_claim(text: str) -> dict | None:
    text = assertion_text(text)
    if not text:
        return None
    text = re.sub(r'^I (?:now|currently) ', 'I ', text, flags=re.I)

    def claim(predicate, value, *, positive=True, exclusive=True, kind='semantic', subject='self', display_subject='User'):
        value = value.strip()
        if predicate in {'residence', 'employer', 'occupation'} or kind == 'preference':
            value = re.sub(r',?\s+now$', '', value, flags=re.I)
        # Do not turn an embedded clause or unspecified placeholder into a value.
        if not value or len(value) > 100 or re.search(r'\b(?:not|never|no longer|it|that|this|because|since|until|with)\b|\|', value, re.I):
            return None
        return {'subject': subject, 'predicate': predicate, 'value': normalized_value(value),
                'positive': positive, 'exclusive': exclusive, 'kind': kind,
                'display_value': value, 'display_subject': display_subject}

    # The explicit "now"/"no longer" forms describe a present change, not history.
    match = re.fullmatch(r"I\s+(?:(do not|don't|no longer)\s+)?(?:(?:now|currently)\s+)?(?:live|reside)\s+in\s+(.+)", text, re.I)
    if match:
        return claim('residence', match[2], positive=match[1] is None)
    match = re.fullmatch(r"I\s+(?:(do not|don't|no longer)\s+)?(?:(?:now|currently)\s+)?work\s+(at|for|as)\s+(.+)", text, re.I)
    if match:
        return claim('occupation' if match[2].lower() == 'as' else 'employer', match[3], positive=match[1] is None)
    match = re.fullmatch(r"My (name|home city|hometown|timezone|time zone|occupation|employer)\s+is\s+(?:(not)\s+)?(?:now\s+)?(.+)", text, re.I)
    if match:
        key = {'home city': 'residence', 'time zone': 'timezone'}.get(match[1].lower(), match[1].lower())
        return claim(key, match[3], positive=match[2] is None)
    match = re.fullmatch(r"My ([\w -]+)'s name is (.+)", text, re.I)
    if match:
        return claim('name', match[2], subject='my:' + normalized_value(match[1]), display_subject='My ' + match[1])
    # Named entities stay distinct from the user and from other named entities.
    match = re.fullmatch(r"([A-Z][\w-]*(?: [A-Z][\w-]*)*) (lives in|works at|works for) (.+)", text)
    if match and match[1].lower() not in {'i', 'he', 'she', 'they', 'we'}:
        return claim('residence' if match[2] == 'lives in' else 'employer', match[3],
                     subject='entity:' + normalized_value(match[1]), display_subject=match[1])
    match = re.fullmatch(r"My (?:favorite|favourite|preferred) ([\w -]+) is (.+)", text, re.I)
    if match:
        return claim('favorite:' + normalized_value(match[1]), match[2], kind='preference')

    # Preferences for response length form a single, explicit dimension.
    positive_tastes = {'like', 'love', 'enjoy', 'prefer'}
    taste_verbs = r"(like|love|enjoy|prefer|dislike|hate|don't like|do not like|can't stand|no longer like)"
    match = re.fullmatch(r'I ' + taste_verbs + r' (short|brief|concise|long|lengthy|detailed) (answers|responses|replies|explanations)(?: now)?', text, re.I)
    if match:
        value = 'concise' if match[2].lower() in {'short', 'brief', 'concise'} else 'detailed'
        return claim('response_length', value, positive=match[1].lower() in positive_tastes, kind='preference')
    match = re.fullmatch(r"I prefer (.+) (?:over|rather than) (.+)", text, re.I)
    if match:
        left, right = normalized_value(match[1]), normalized_value(match[2])
        if left != right:
            result = claim('choice:' + '|'.join(sorted((left, right))), match[1], kind='preference')
            if result:
                result['alternative'] = match[2].strip()
            return result
    match = re.fullmatch(r'I ' + taste_verbs + r' (.+)', text, re.I)
    if match:
        return claim('taste', match[2], positive=match[1].lower() in positive_tastes,
                     exclusive=False, kind='preference')
    match = re.fullmatch(r"I(?: am|'m) (not )?allergic to (.+)", text, re.I)
    if match:
        return claim('allergy', match[2], positive=match[1] is None, exclusive=False, kind='preference')
    return None


def equivalent(left: dict, right: dict) -> bool:
    return all(left.get(key) == right.get(key) for key in ('subject', 'predicate', 'value', 'positive', 'exclusive'))


def contradicts(left: dict, right: dict) -> bool:
    if (left['subject'], left['predicate']) != (right['subject'], right['predicate']):
        return False
    if left['value'] == right['value']:
        return left['positive'] != right['positive']
    return left['exclusive'] and right['exclusive'] and left['positive'] and right['positive']


def claim_summary(claim: dict) -> str:
    subject, value, positive = claim['display_subject'], claim['display_value'], claim['positive']
    predicate = claim['predicate']
    if predicate == 'residence':
        return f"{subject} {'lives' if positive else 'does not live'} in {value}."
    if predicate == 'employer':
        return f"{subject} {'works' if positive else 'does not work'} at {value}."
    if predicate == 'taste':
        return f"{subject} {'likes' if positive else 'does not like'} {value}."
    if predicate == 'allergy':
        return f"{subject} is {'allergic' if positive else 'not allergic'} to {value}."
    if predicate == 'response_length':
        return f"{subject} {'prefers' if positive else 'does not prefer'} {value} responses."
    if predicate.startswith('choice:'):
        return f"{subject} prefers {value} over {claim['alternative']}."
    label = 'favorite ' + predicate[9:] if predicate.startswith('favorite:') else predicate
    return f"{subject}'s {label} is {'' if positive else 'not '}{value}."
