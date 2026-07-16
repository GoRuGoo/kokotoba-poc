from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class Memory:
    id: int | None
    text: str
    embedding: list[float]
    tags: list[str] = field(default_factory=list)
    importance: float = 0.0
    created_at: datetime | None = None
