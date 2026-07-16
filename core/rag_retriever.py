from __future__ import annotations

from dataclasses import dataclass

from ai.embedding import EmbeddingService
from database.repository import Repository
from models.memory import Memory
from utils.cosine_similarity import CosineSimilarity


@dataclass(slots=True)
class RankedMemory:
    memory: Memory
    score: float


class RAGRetriever:
    def __init__(self, repository: Repository, embedding_service: EmbeddingService, top_k: int = 5) -> None:
        self.repository = repository
        self.embedding_service = embedding_service
        self.top_k = top_k

    def retrieve(self, question: str) -> list[RankedMemory]:
        question_embedding = self.embedding_service.embed(question)
        memories = self.repository.list_memories()

        ranked_memories = [
            RankedMemory(memory=memory, score=self._score_memory(
                question, question_embedding, memory))
            for memory in memories
        ]
        ranked_memories.sort(key=lambda item: item.score, reverse=True)
        return ranked_memories[: self.top_k]

    def _score_memory(self, question: str, question_embedding: list[float], memory: Memory) -> float:
        score = CosineSimilarity.calculate(
            question_embedding, memory.embedding)
        lowered_question = question.lower()
        lowered_memory = memory.text.lower()
        memory_tags = {tag.lower() for tag in memory.tags}

        symptom_question = any(keyword in lowered_question for keyword in [
                               "症状", "どんな症状", "痛み", "体調", "具合", "しんど", "発作"])
        medical_memory = "medical_history" in memory_tags or "既往歴" in lowered_memory or "持病" in lowered_memory

        if symptom_question and medical_memory:
            score += 1.0

        if any(keyword in lowered_question for keyword in ["既往歴", "持病", "病歴", "治療", "通院"]):
            if medical_memory:
                score += 0.5

        if any(tag in lowered_question for tag in memory_tags):
            score += 0.25

        return score
