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

        # SQLiteではデフォルトで無効なので有効化
        conn.execute("PRAGMA foreign_keys = ON")

        return conn

    def initialize(self):
        with self.connect() as conn:
            # 長期記憶
            conn.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                summary TEXT NOT NULL,
                source_text TEXT NOT NULL,

                place_name TEXT,
                latitude REAL,
                longitude REAL,

                speaker TEXT,
                event_time DATETIME,

                importance REAL DEFAULT 0.5,

                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                modified_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # タグ辞書
            conn.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                name TEXT NOT NULL UNIQUE,
                description TEXT
            )
            """)

            # 長期記憶とタグの対応
            conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_tag_map (
                memory_id INTEGER NOT NULL,
                tag_id INTEGER NOT NULL,

                PRIMARY KEY (memory_id, tag_id),

                FOREIGN KEY (memory_id)
                    REFERENCES long_term_memory(id)
                    ON DELETE CASCADE,

                FOREIGN KEY (tag_id)
                    REFERENCES tags(id)
                    ON DELETE CASCADE
            )
            """)

            conn.commit()
