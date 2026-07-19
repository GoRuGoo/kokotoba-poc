"""長期記憶を検索して回答を生成するRAG処理。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm.llm_client import LLMClient

if TYPE_CHECKING:
    from database.init import DatabaseManager


class RAG:
    def __init__(
        self,
        llm_client: LLMClient,
        db_manager: DatabaseManager,
    ) -> None:
        self.llm_client = llm_client
        self.db_manager = db_manager

    def generate_rag_response(self, user_input: str, user_location: str) -> None:
        expanded_query = self._query_expansion(user_input, user_location)
        print(f"Expanded Query: {expanded_query}")

    def _query_expansion(self, user_input: str, user_location: str) -> str:
        """抽象的な質問と場所を、長期記憶検索向けの文字列へ拡張する。"""
        if not user_input.strip():
            raise ValueError("user_input must not be empty or whitespace only")
        if not user_location.strip():
            raise ValueError(
                "user_location must not be empty or whitespace only")

        prompt = f"""
あなたは、ユーザーの長期記憶をベクトル検索するためのクエリ拡張器です。
次の「質問」と「現在地」から、過去の記録の検索にヒットしそうな単語を5個出力してください。

<question>
{user_input}
</question>

<location>
{user_location}
</location>

ルール:
- 質問への回答はせず、検索に使う文字列だけを出力する
- スペース区切りで出力すること
- 元の質問文と現在地の名称を必ず含める
- 現在地を病院など特定の施設種別に決めつけない
- 施設名から考えられる種別、行動、出来事、話題、言い換え、関連語を補う
- 確認できない具体的な体験、人物名、日時、症状、商品などを捏造しない
- 見出し、箇条書き、説明、引用符は付けず、1〜3文の自然な検索文にする
""".strip()

        expanded_query = self.llm_client.generate(prompt).strip()
        if not expanded_query:
            raise RuntimeError("LLM returned an empty expanded query")

        return expanded_query
