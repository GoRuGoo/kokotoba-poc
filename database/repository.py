from __future__ import annotations

import json
from datetime import datetime, timezone

from database.database import Database
from models.conversation import Conversation
from models.memory import Memory


class Repository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def save_conversation_message(self, session_id: str, role: str, content: str, created_at: datetime) -> None:
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO conversation (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, created_at.isoformat()),
            )
            connection.commit()

    def save_conversation(self, conversation: Conversation) -> None:
        for message in conversation.messages:
            self.save_conversation_message(
                conversation.session_id, message.role, message.content, message.created_at)

    def list_memories(self) -> list[Memory]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT id, text, embedding, tags, importance, created_at FROM memory ORDER BY id ASC"
            ).fetchall()

        memories: list[Memory] = []
        for row in rows:
            memories.append(
                Memory(
                    id=row["id"],
                    text=row["text"],
                    embedding=json.loads(row["embedding"]),
                    tags=json.loads(row["tags"]),
                    importance=float(row["importance"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            )
        return memories

    def save_memory(self, memory: Memory) -> None:
        created_at = memory.created_at or datetime.now(timezone.utc)
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO memory (text, embedding, tags, importance, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    memory.text,
                    json.dumps(memory.embedding),
                    json.dumps(memory.tags, ensure_ascii=False),
                    memory.importance,
                    created_at.isoformat(),
                ),
            )
            connection.commit()
