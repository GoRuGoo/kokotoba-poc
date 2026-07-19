from core.chat_manager import ChatManager
from database.init import DatabaseManager
from llm import LLMClient, LiteRTLMClient


def main() -> None:
    db_manager = DatabaseManager()
    chat_manager = ChatManager(db_manager)

    llm_manager: LLMClient = LiteRTLMClient(
        "models/gemma-4-E2B-it.litertlm"
    )

    llm_manager.start()

    # テーブル作成
    db_manager.initialize()
    print("データベースの初期化が完了しました。")
    db_manager.insert_demo_data()
    print("デモデータの挿入が完了しました。")

    while True:
        # 店員・医者からの入力と固定の地名を受け取る
        response = llm_manager.generate(
            "日本の首都は?一文で完結に答えてください。アスタリスクなどの特殊記号はいりません。")
        print(response)
        user_input = input("Doctor: ").strip()
        user_location_input = input("Your location: ").strip()

        if user_input.lower() == "exit":
            chat_manager.finish_session()
            break

        cards = chat_manager.handle_user_input(user_input, user_location_input)


if __name__ == "__main__":
    main()
