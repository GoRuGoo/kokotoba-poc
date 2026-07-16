from __future__ import annotations

import json
from dataclasses import dataclass
import re

from ai.embedding import EmbeddingService
from ai.llm import LLM
from database.repository import Repository
from models.conversation import Conversation
from models.memory import Memory


@dataclass(slots=True)
class ConsolidatedMemory:
    memory: str
    tags: list[str]
    importance: float


class MemoryConsolidator:
    def __init__(self, llm: LLM, embedding_service: EmbeddingService, repository: Repository) -> None:
        self.llm = llm
        self.embedding_service = embedding_service
        self.repository = repository

    def consolidate(self, conversation: Conversation) -> Memory | None:
        payload = self._extract_memory(conversation)

        if payload is None:
            prompt = self._build_prompt(conversation)
            response = self.llm.generate(prompt)
            payload = self._parse_response(response)

        if payload is None:
            return None

        memory = Memory(
            id=None,
            text=payload.memory,
            embedding=self.embedding_service.embed(payload.memory),
            tags=payload.tags,
            importance=payload.importance,
        )
        self.repository.save_memory(memory)
        return memory

    def _build_prompt(self, conversation: Conversation) -> str:
        lines = [
            f"{message.role}: {message.content}" for message in conversation.messages]
        return "\n".join(
            [
                "以下の会話から長期保存すべき情報をJSONで1件抽出してください。",
                *lines,
                '出力形式: {"memory":"...","tags":["..."],"importance":0.0}',
            ]
        )

    def _extract_memory(self, conversation: Conversation) -> ConsolidatedMemory | None:
        user_texts = [
            message.content.strip()
            for message in conversation.messages
            if message.role == "user" and message.content.strip()
        ]

        if not user_texts:
            return None

        source_text = user_texts[-1]
        medical_history = self._extract_medical_history(source_text)
        if medical_history is not None:
            return ConsolidatedMemory(
                memory=f"既往歴: {medical_history}",
                tags=["medical_history", medical_history],
                importance=0.95,
            )

        if len(source_text) <= 40:
            return ConsolidatedMemory(
                memory=f"会話メモ: {source_text}",
                tags=["conversation"],
                importance=0.5,
            )

        return None

    def _extract_medical_history(self, text: str) -> str | None:
        normalized = text.lower().replace("ガン", "がん")
        disease_patterns = [
            (r"(がん|癌)", "がん"),
            (r"糖尿病", "糖尿病"),
            (r"高血圧", "高血圧"),
            (r"喘息", "喘息"),
            (r"心不全", "心不全"),
            (r"脳梗塞", "脳梗塞"),
            (r"認知症", "認知症"),
            (r"てんかん", "てんかん"),
            (r"パーキンソン", "パーキンソン病"),
            (r"慢性腎臓病|ckd", "慢性腎臓病"),
        ]

        for pattern, label in disease_patterns:
            if re.search(pattern, normalized):
                return label

        history_markers = ["既往歴", "持病", "病歴", "治療中", "通院中", "手術歴", "入院歴"]
        if any(marker in text for marker in history_markers):
            cleaned = re.sub(
                r".*?(既往歴|持病|病歴|治療中|通院中|手術歴|入院歴)[:：\s]*", "", text)
            cleaned = cleaned.strip(" 。．、,，\n\t")
            return cleaned or None

        if re.fullmatch(r"[ぁ-んァ-ヶ一-龥ー々]+", text):
            return text

        return None

    def _parse_response(self, response: str) -> ConsolidatedMemory | None:
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            return None

        memory = str(payload.get("memory", "")).strip()
        tags = [str(tag)
                for tag in payload.get("tags", []) if str(tag).strip()]
        importance = float(payload.get("importance", 0.0))

        if not memory:
            return None

        return ConsolidatedMemory(memory=memory, tags=tags, importance=importance)
