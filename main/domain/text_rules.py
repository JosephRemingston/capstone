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


def durable_fact(text: str) -> bool:
    """Recognize useful declarative conversational facts missed by narrow claims."""
    text = normalize(text)
    if uncertain(text, quoted_objects=True):
        return False
    return matches(
        r"^(?:please remember(?: that)?|remember(?: that)?)\s+\S|"
        r"^(?:i am|i'm)\s+(?:responsible for|working on|part of|based in)\s+\S|"
        r"^(?:we|our (?:team|project|company))\s+(?:use|uses|work|works|is|are|has|have)\s+\S|"
        r"^(?:the|this|that)\s+[\w -]{2,50}\s+(?:uses|requires|belongs to|depends on|is owned by)\s+\S", text)


_PHATIC = (
    r"hi|hello|hey|hiya|yo|howdy|good (?:morning|afternoon|evening|night)|"
    r"thanks?(?: (?:a lot|so much))?|thank you(?: (?:so much|very much))?|thx|ty|cheers|"
    r"much appreciated|(?:i )?appreciate (?:it|that)|you'?re welcome|"
    r"ok(?:ay)?|k+|sure|yes|yeah|yep|yup|no|nope|nah|alright|all right|fine|cool|great|nice|perfect|awesome|wow|"
    r"got it|gotcha|understood|noted|will do|sounds (?:good|great)|(?:that )?makes sense|fair enough|"
    r"i see|i agree|i understand|i (?:get|got) it|i know|no idea|i have no idea|i (?:don't|do not) know|"
    r"(?:i'?ll|i will)(?: (?:try|do) (?:that|it|this))?|(?:i )?(?:have|gotta|got) to go|"
    r"(?:i'?m|i am) (?:good|fine|ok|okay|great|back|done|here|ready|sorry|glad|confused|lost|not sure)|"
    r"lol|lmao|rofl|ha(?:ha)+|hehe|"
    r"bye|goodbye|see you(?: later)?|see ya|talk (?:to you )?later|ttyl|"
    r"never ?mind|nvm|forget (?:it|about it)|no worries|no problem|np|my bad|"
    r"sorry|pardon(?: me)?|huh|what|come again|excuse me|please|"
    r"(?:(?:can|could|would) you (?:please )?)?(?:say|repeat) (?:that|it)|"
    r"that'?s (?:great|good|cool|helpful|perfect|awesome|fine|ok|okay|interesting|it)|"
    r"now|then|again|anyway|so|really|too|for (?:now|today)"
)
# First person that carries no information about the user: interrogative inversion,
# requests addressed to the assistant, and reactions to what the assistant said.
_NON_DISCLOSING = (
    r"\b(?:can|could|would|will|should|shall|may|might|must|do|does|did|am|are|have|has|was|were) (?:i|we)\b|"
    r"\b(?:tell|show|give|help|let|send|remind) (?:me|us)\b|"
    r"\b(?:i|we) (?:was|were|am|'m|are|'re) (?:just )?(?:wondering|curious)\b|"
    r"\b(?:i|we)(?:'d| would) (?:really )?(?:like|love|appreciate)"
    r"(?: it if| (?:you|your help|some help)| to (?:know|learn|ask|hear|understand|find out))\b|"
    r"\b(?:i|we) (?:want|wanted|need|needed) to (?:know|ask|learn|understand|find out)\b|"
    r"\b(?:i|we) (?:really )?(?:love|like) (?:it|that|this|these|those|them|the idea|your)\b|"
    r"\b(?:i|we) (?:didn't|did not|don't|do not) (?:catch|get|follow|hear|understand)\b|"
    r"\b(?:i|we) (?:get|got) it\b|"
    r"\b(?:i|we) (?:see|agree|understand|guess|mean|think|suppose|hope|know|wonder|wondered|believe|appreciate)\b"
)
# Asking the assistant to recall a stored fact ("What was my dog's name again?") discloses nothing
# new. Only short whole questions match, so "What's my best option given I'm moving?" still counts.
_RECALL_QUESTION = re.compile(
    r"(?:^|(?<=[.!?;]))\s*(?:\w+,\s*)?(?:what|where|when|who|which)(?:'s|\s+(?:is|was|are|were))\s+(?:my|our)\s+"
    r"(?:[\w'-]+\s+){0,4}?[\w'-]+(?:\s+(?:again|called|named|now))?\s*\?")
# An uppercase roman numeral I is not the pronoun: "World War I", "type I diabetes",
# "Elizabeth I." Only matched after a numbering keyword or a capitalised word that ends
# the clause, so "Yesterday I went..." keeps its pronoun.
_ROMAN_I = re.compile(
    r"\b(?:[Tt]ype|[Pp]art|[Pp]hase|[Cc]hapter|[Ll]evel|[Gg]rade|[Ss]tage|[Cc]lass|[Aa]ct|[Bb]ook|[Vv]olume|[Ww]ar)\s+I\b"
    r"|\b[A-Z][a-z]+\s+I(?=\s*(?:[.:!?)]|$))"
)


def phatic(text: str) -> bool:
    """Greetings, thanks, acknowledgements and repair requests with nothing else in them."""
    words = ' '.join(re.sub(r"[^\w\s']", ' ', normalize(text)).split())
    return not words or re.fullmatch(rf"(?:(?:{_PHATIC})(?: |$))+", words + ' ') is not None


def self_disclosure(text: str) -> bool:
    """The speaker states something about themselves (circumstances, plans, possessions,
    events), possibly inside a question or a request. Quoted speech is somebody else's."""
    text = normalize(_ROMAN_I.sub(' ', text))
    if phatic(text):
        return False
    if not text.startswith(('"', "'")):
        text = re.sub(r'"[^"\n]*"', ' ', text)
    text = _RECALL_QUESTION.sub(' ', text)
    text = re.sub(_NON_DISCLOSING, ' ', text)
    return matches(r"\b(?:i|we)\b|\b(?:my|our)\s+\w", text)
