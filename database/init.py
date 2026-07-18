import sqlite3
from pathlib import Path


class DatabaseManager:
    def __init__(self):
        self.path = Path("data/app.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.initialize()

    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self):
        with self.connect() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS short_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_input TEXT NOT NULL,
                created_at NOT NULL DEFAULT CURRENT_TIMESTAMP
                modified_at NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary TEXT NOT NULL,
                source_text TEXT NOT NULL,
                created_at NOT NULL DEFAULT CURRENT_TIMESTAMP
                modified_at NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """)
