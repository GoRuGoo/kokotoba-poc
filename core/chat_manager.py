from __future__ import annotations

from ai.embedding import EmbeddingService
from ai.llm import LLM, MockLLM
from config.settings import AppSettings, get_settings
from core.card_generator import CardGenerator
from core.memory_consolidator import MemoryConsolidator
from core.rag_retriever import RAGRetriever
from database.database import Database
from database.repository import Repository
from models.conversation import Conversation
from models.card import SuggestionCard


class ChatManager:
    def __init__(
        self,
        settings: AppSettings | None = None,
        llm: LLM | None = None,
        embedding_service: EmbeddingService | None = None,
        repository: Repository | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.database = Database(self.settings.database_path)
        self.repository = repository or Repository(self.database)
        self.embedding_service = embedding_service or EmbeddingService(
            dimension=self.settings.embedding_dimension)
        self.llm = llm or MockLLM()
        self.memory_llm = MockLLM()
        self.retriever = RAGRetriever(
            self.repository, self.embedding_service, top_k=self.settings.top_k)
        self.card_generator = CardGenerator(None)
        self.memory_consolidator = MemoryConsolidator(
            self.memory_llm, self.embedding_service, self.repository)
        self.conversation = Conversation()

    def handle_user_input(self, user_input: str) -> list[SuggestionCard]:
        self.conversation.add_message("user", user_input)
        memories = self.retriever.retrieve(user_input)
        cards = self.card_generator.generate(user_input, memories)
        assistant_summary = " / ".join(card.title for card in cards)
        self.conversation.add_message("assistant", assistant_summary)
        return cards

    def finish_session(self) -> None:
        self.repository.save_conversation(self.conversation)
        self.memory_consolidator.consolidate(self.conversation)
