from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class Message:
    role: str
    content: str
    created_at: datetime
