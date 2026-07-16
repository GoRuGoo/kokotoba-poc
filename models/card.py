from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SuggestionCard:
    title: str
    subtitle: str
    reason: str
    confidence: float
    tags: list[str] = field(default_factory=list)
