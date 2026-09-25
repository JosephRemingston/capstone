"""Rule-based memory classification."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import MemoryCategory, MemoryInput
from .text_rules import event_update, matches, preference_constraint, negated_task


def _contains_any(text: str, terms: tuple[str, ...], *, legacy: bool = False) -> bool:
    if legacy:
        return any(term in text for term in terms)
    return any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text) for term in terms)


@dataclass(slots=True)
class MemoryClassifier:
    """Deterministic classifier with an easy replacement point for ML/LLM models."""

    legacy: bool = False  # Frozen preprocessing for the existing Hippocorpus artifact.

    temporary_terms: tuple[str, ...] = (
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "cool",
        "lol",
    )
    preference_terms: tuple[str, ...] = (
        "i like",
        "i prefer",
        "i love",
        "favorite",
        "favourite",
        "i dislike",
        "i hate",
        "don't like",
        "do not like",
    )
    task_terms: tuple[str, ...] = (
        "remind me",
        "todo",
        "to-do",
        "follow up",
        "schedule",
        "book",
        "call me",
        "deadline",
        "due",
        "need to",
        "please do",
    )
    procedural_terms: tuple[str, ...] = (
        "how to",
        "steps",
        "procedure",
        "workflow",
        "when i",
        "whenever i",
        "always",
        "first",
        "then",
        "finally",
    )
    episodic_patterns: tuple[re.Pattern[str], ...] = field(
        default_factory=lambda: (
            re.compile(r"\b(yesterday|today|tomorrow|last|next)\b", re.I),
            re.compile(r"\b(on|at)\s+\d{1,2}(:\d{2})?\b", re.I),
            re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
            re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I),
        )
    )
    semantic_patterns: tuple[re.Pattern[str], ...] = field(
        default_factory=lambda: (
            re.compile(r"\b(my|our)\s+[\w -]{2,40}\s+(is|are|was|were)\b", re.I),
            re.compile(r"\bi\s+(am|work|live|study|have|own)\b", re.I),
            re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"),
        )
    )

    def classify(self, memory_input: MemoryInput) -> MemoryCategory:
        text = memory_input.content.strip()
        if not self.legacy:
            text = text.replace("’", "'")
        lowered = text.lower()

        if not text:
            return MemoryCategory.TEMPORARY

        normalized = re.sub(r"[^\w\s'-]", "", lowered).strip()
        if normalized in self.temporary_terms or (self.legacy and len(normalized.split()) <= 2):
            return MemoryCategory.TEMPORARY

        if not self.legacy:
            if matches(r"\b(?:don't|do not|no longer)\s+(?:need|want)\s+to\b", lowered):
                return MemoryCategory.EPISODIC
            if negated_task(lowered):
                return MemoryCategory.TASK
            if preference_constraint(lowered):
                return MemoryCategory.PREFERENCE
            # An explicit new request takes precedence over an event it mentions.
            if matches(r"\b(?:remind me|need to|please do|todo|to-do)\b", lowered):
                return MemoryCategory.TASK
            if event_update(lowered):
                return MemoryCategory.EPISODIC
        task_terms = self.task_terms if self.legacy else tuple(term for term in self.task_terms if term not in {"book", "schedule"})
        if _contains_any(lowered, task_terms, legacy=self.legacy):
            return MemoryCategory.TASK

        if not self.legacy and matches(r"^(?:please\s+)?(?:book|schedule|buy|submit|send|call|review|cancel|finish|pay)\b\s+\S", lowered):
            return MemoryCategory.TASK

        if _contains_any(lowered, self.preference_terms, legacy=self.legacy):
            return MemoryCategory.PREFERENCE

        if _contains_any(lowered, self.procedural_terms, legacy=self.legacy):
            return MemoryCategory.PROCEDURAL

        if any(pattern.search(text) for pattern in self.episodic_patterns):
            return MemoryCategory.EPISODIC

        if any(pattern.search(text) for pattern in self.semantic_patterns):
            return MemoryCategory.SEMANTIC

        return MemoryCategory.TEMPORARY
