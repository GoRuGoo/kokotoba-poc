from __future__ import annotations

from core.rag_retriever import RankedMemory
from models.conversation import Conversation


class PromptBuilder:
    def build(self, conversation: Conversation, memories: list[RankedMemory]) -> str:
        conversation_lines = [
            f"{message.role}: {message.content}" for message in conversation.messages]
        memory_lines = [
            f"- {item.memory.text} (tags: {', '.join(item.memory.tags)})" for item in memories]

        parts = [
            "あなたは発語障害支援アプリのアシスタントです。",
            "長期記憶に既往歴や持病があれば、それを前提に推測して返答してください。",
            "症状をただ聞き返すだけで終わらせず、考えられる症状や確認ポイントを短く示してください。",
            "参考メモリ:",
            *memory_lines,
            "会話履歴:",
            *conversation_lines,
            "支援しやすい短い日本語で返答してください。",
        ]
        return "\n".join(parts)
