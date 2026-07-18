from database import DatabaseManager
from memory.memory_consolidator import MemoryConsolidator, ShortTermMemoryRecord, Location
from time import datetime


class ChatManager:

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        pass

    def handle_user_input(self, user_input: str, user_location_input: str) -> list[str]:
        # 短期記憶に会話内容を保存
        self.memory_consolidator.record_short_term_memory(
            ShortTermMemoryRecord(
                conversation_text=user_input,
                timestamp=datetime.now(),
                location=Location(
                    latitude=0.0,          # 必要に応じて実際の緯度
                    longitude=0.0,         # 必要に応じて実際の経度
                    place_name=user_location_input,
                ),
                speaker="user",
            )
        )

    def finish_session(self) -> None:
        return

    def _build_card(self, user_input: str, user_location_input: str) -> str:
        return f"{user_input} / {user_location_input}"
