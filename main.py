from core.chat_manager import ChatManager
from database.init import DatabaseManager


def main() -> None:
    chat_manager = ChatManager()
    db_manager = DatabaseManager()

    # テーブル作成
    db_manager.initialize()

    while True:
        # 店員・医者からの入力と固定の地名を受け取る
        user_input = input("Doctor: ").strip()
        user_location_input = input("Your location: ").strip()

        if user_input.lower() == "exit":
            chat_manager.finish_session()
            break

        cards = chat_manager.handle_user_input(user_input, user_location_input)


if __name__ == "__main__":
    main()
