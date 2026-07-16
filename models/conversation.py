from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable
from uuid import uuid4

from models.message import Message


@dataclass
class Conversation:
    session_id: str = field(default_factory=lambda: str(uuid4()))
    messages: list[Message] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> Message:
        message = Message(role=role, content=content,
                          created_at=datetime.now(timezone.utc))
        self.messages.append(message)
        return message

    def extend(self, messages: Iterable[Message]) -> None:
        self.messages.extend(messages)
