"""長期記憶を検索して回答を生成するRAG処理。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from database.long_term_memory_searcher import LongTermMemorySearcher
from llm.llm_client import LLMClient
from util.embedding import Embedding

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
        self.embedding = Embedding()
        self.long_term_memory_searcher = LongTermMemorySearcher(db_manager)

    def generate_rag_response(
        self,
        user_input: str,
        user_location: str,
    ) -> str:
        expanded_query = self._query_expansion(user_input, user_location)
        query_embedding = self.embedding.embed(expanded_query)
        long_term_memory_result = self.long_term_memory_searcher.search(
            query_embedding
        )

        card_prompt = f"""
あなたは、発語障害のあるユーザーがボタンを押すだけで相手の質問に答えられるよう、
短く分かりやすい発話カードの候補を作成する支援者です。
次の質問、現在地、関連する過去の記憶を参考に、ユーザー本人がそのまま発言できる
回答候補を作成してください。

<question>
{user_input}
</question>

<location>
{user_location}
</location>

<long_term_memory>
{long_term_memory_result or "該当する長期記憶なし"}
</long_term_memory>

ルール:
- 過去の記憶は候補を考えるための参考情報であり、現在も同じ状態だと断定しない
- long_term_memoryは過去の情報が必要なときに参考程度に使用すること。単純な質問にはあなたの回答だけで十分であり、過去の記憶を無理に使う必要はない
- 質問への直接的な回答になる、一人称視点の短い候補を2〜5個作る
- なるべく質問の意図に沿った候補を作る
- 過去の記憶から考えられる内容を優先し、無関係な内容を捏造しない
- 医療診断や確定していない事実を作らない
- 情報が不足している場合は「わからない」などの安全な候補を含める
- 1候補はボタンに表示できる短さにし、候補内にスペースを入れない
- 候補と候補の間は半角スペース1個で区切る
- 見出し、番号、箇条書き、説明、引用符、句読点、改行は出力しない

出力例:
頭が痛い まだ熱がある だいぶ良くなった わからない
""".strip()

        generated_cards = self.llm_client.generate(card_prompt).strip()
        if not generated_cards:
            raise RuntimeError("LLM returned empty card candidates")

        card_candidates = " ".join(generated_cards.split())
        print(f"カード候補: {card_candidates}")
        return card_candidates

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
- 元の質問文と現在地の文字列をそのまま出力しない
- 現在地を病院など特定の施設種別に決めつけない
- 施設名から考えられる種別、行動、出来事、話題、言い換え、関連語を補う
- 確認できない具体的な体験、人物名、日時、症状、商品などを捏造しない
- 見出し、箇条書き、説明、引用符は付けず、1〜3文の自然な検索文にする
""".strip()

        expanded_query = self.llm_client.generate(prompt).strip()
        if not expanded_query:
            raise RuntimeError("LLM returned an empty expanded query")

        return expanded_query
