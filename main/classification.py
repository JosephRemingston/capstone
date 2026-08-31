"""Rule-based memory classification."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import MemoryCategory, MemoryInput


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


@dataclass(slots=True)
class MemoryClassifier:
    """Deterministic classifier with an easy replacement point for ML/LLM models."""

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
        lowered = text.lower()

        if not text:
            return MemoryCategory.TEMPORARY

        normalized = re.sub(r"[^\w\s'-]", "", lowered).strip()
        if normalized in self.temporary_terms or len(normalized.split()) <= 2:
            return MemoryCategory.TEMPORARY

        if _contains_any(lowered, self.task_terms):
            return MemoryCategory.TASK

        if _contains_any(lowered, self.preference_terms):
            return MemoryCategory.PREFERENCE

        if _contains_any(lowered, self.procedural_terms):
            return MemoryCategory.PROCEDURAL

        if any(pattern.search(text) for pattern in self.episodic_patterns):
            return MemoryCategory.EPISODIC

        if any(pattern.search(text) for pattern in self.semantic_patterns):
            return MemoryCategory.SEMANTIC

        return MemoryCategory.TEMPORARY
