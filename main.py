from core.chat_manager import ChatManager


def _print_cards(cards) -> None:
    print("Cards:")
    for index, card in enumerate(cards, start=1):
        print(f"[{index}] {card.title}")
        if card.subtitle:
            print(f"    {card.subtitle}")
        print(f"    理由: {card.reason}")
        print(f"    信頼度: {card.confidence:.2f}")


def _choose_card(cards):
    if not cards:
        return None

    while True:
        choice = input("選ぶ番号を入力してください(1-3, Enterでスキップ): ").strip()
        if choice == "":
            return None
        if not choice.isdigit():
            print("数字で入力してください。")
            continue

        index = int(choice) - 1
        if 0 <= index < len(cards):
            return cards[index]

        print("範囲内の番号を入力してください。")


def main() -> None:
    chat_manager = ChatManager()

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            chat_manager.finish_session()
            break

        cards = chat_manager.handle_user_input(user_input)
        _print_cards(cards)
        selected_card = _choose_card(cards)
        if selected_card is not None:
            print(f"選択: {selected_card.title}")
            print(f"発話候補: {selected_card.subtitle}")


if __name__ == "__main__":
    main()
