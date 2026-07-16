from __future__ import annotations

import json
import re

from ai.llm import LLM
from core.rag_retriever import RankedMemory
from models.card import SuggestionCard
from models.memory import Memory


class CardGenerator:
    def __init__(self, llm: LLM | None = None) -> None:
        self.llm = llm

    def generate(self, user_input: str, memories: list[RankedMemory]) -> list[SuggestionCard]:
        if self.llm is not None:
            cards = self._generate_with_llm(user_input, memories)
            if cards:
                return cards

        return self._generate_fallback(user_input, memories)

    def _generate_with_llm(self, user_input: str, memories: list[RankedMemory]) -> list[SuggestionCard]:
        prompt = self._build_prompt(user_input, memories)
        try:
            response = self.llm.generate(prompt)
        except Exception:
            return []

        return self._parse_cards(response)

    def _build_prompt(self, user_input: str, memories: list[RankedMemory]) -> str:
        memory_lines = [
            f"- {item.memory.text} (tags: {', '.join(item.memory.tags)})" for item in memories
        ]
        return "\n".join(
            [
                "あなたは発語支援アプリです。音声認識結果と長期記憶から、使用者が本当に言いたい内容をカード候補として3件出力してください。",
                "出力はJSONのみです。形式: {\"cards\":[{\"title\":\"...\",\"subtitle\":\"...\",\"reason\":\"...\",\"confidence\":0.0,\"tags\":[\"...\"]}]}",
                f"音声認識結果: {user_input}",
                "長期記憶:",
                *memory_lines,
                "カードは、短く自然な日本語で、使用者がそのまま選べる文にしてください。",
            ]
        )

    def _parse_cards(self, response: str) -> list[SuggestionCard]:
        candidates = [response.strip()]
        json_text = self._extract_json_text(response)
        if json_text and json_text not in candidates:
            candidates.append(json_text)

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            cards_payload = payload.get(
                "cards") if isinstance(payload, dict) else None
            if not isinstance(cards_payload, list):
                continue

            cards: list[SuggestionCard] = []
            for item in cards_payload:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip()
                subtitle = str(item.get("subtitle", "")).strip()
                reason = str(item.get("reason", "")).strip()
                confidence = float(item.get("confidence", 0.0))
                tags = [str(tag).strip()
                        for tag in item.get("tags", []) if str(tag).strip()]
                if not title:
                    continue
                cards.append(
                    SuggestionCard(
                        title=title,
                        subtitle=subtitle or title,
                        reason=reason or "LLMが候補を生成しました。",
                        confidence=confidence,
                        tags=tags,
                    )
                )

            if cards:
                return cards[:3]

        return []

    def _extract_json_text(self, response: str) -> str | None:
        start = response.find("{")
        end = response.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return response[start: end + 1]

    def _generate_fallback(self, user_input: str, memories: list[RankedMemory]) -> list[SuggestionCard]:
        disease = self._infer_disease(memories)
        lowered_input = user_input.lower()

        if disease and self._looks_like_symptom_question(lowered_input):
            return [
                SuggestionCard(
                    title=f"{disease}の症状が心配です",
                    subtitle=f"{disease}で出やすい症状を知りたい",
                    reason="既往歴をもとに、症状確認の意図を推測しました。",
                    confidence=0.97,
                    tags=["medical_history", "symptom"],
                ),
                SuggestionCard(
                    title="受診の目安を知りたい",
                    subtitle="今すぐ病院に行くべきか確認したい",
                    reason="症状の不安から受診判断のカードを追加しました。",
                    confidence=0.82,
                    tags=["triage", "medical"],
                ),
                SuggestionCard(
                    title="気になる症状を伝えたい",
                    subtitle="痛みや変化を短く整理したい",
                    reason="発話支援としてそのまま使える表現を用意しました。",
                    confidence=0.74,
                    tags=["communication"],
                ),
            ]

        if disease:
            return [
                SuggestionCard(
                    title=f"{disease}について相談したい",
                    subtitle=f"{disease}に関する不安を伝えたい",
                    reason="既往歴から相談内容を推測しました。",
                    confidence=0.92,
                    tags=["medical_history"],
                ),
                SuggestionCard(
                    title="症状を整理して伝えたい",
                    subtitle="今の状態を短く説明したい",
                    reason="会話カードとして使いやすい表現にしています。",
                    confidence=0.76,
                    tags=["communication"],
                ),
                SuggestionCard(
                    title="受診の必要性を知りたい",
                    subtitle="病院に行くべきか確認したい",
                    reason="相談の次の行動を推測して補いました。",
                    confidence=0.71,
                    tags=["triage"],
                ),
            ]

        text = user_input.strip()
        if self._looks_like_symptom_question(lowered_input):
            return [
                SuggestionCard(
                    title="症状について相談したい",
                    subtitle="今の体調を知りたい",
                    reason="症状確認の意図を推測しました。",
                    confidence=0.84,
                    tags=["symptom"],
                ),
                SuggestionCard(
                    title="受診の目安を知りたい",
                    subtitle="病院に行くか判断したい",
                    reason="相談内容として自然な候補を補いました。",
                    confidence=0.79,
                    tags=["triage"],
                ),
                SuggestionCard(
                    title="気になることを整理したい",
                    subtitle="伝えたい内容を短くまとめたい",
                    reason="発話支援として使いやすいカードにしています。",
                    confidence=0.69,
                    tags=["communication"],
                ),
            ]

        return [
            SuggestionCard(
                title=f"{text}について相談したい",
                subtitle="伝えたい内容を整理したい",
                reason="入力文をもとに最も自然なカードを作成しました。",
                confidence=0.65,
                tags=["general"],
            ),
            SuggestionCard(
                title="症状を確認したい",
                subtitle="体調の不安を伝えたい",
                reason="発話支援の定番カードを補いました。",
                confidence=0.6,
                tags=["general"],
            ),
            SuggestionCard(
                title="受診について相談したい",
                subtitle="病院に行くべきか知りたい",
                reason="次の行動を選べるようにしています。",
                confidence=0.58,
                tags=["general"],
            ),
        ]

    def _infer_disease(self, memories: list[RankedMemory]) -> str | None:
        for item in memories:
            memory = item.memory
            if "medical_history" in {tag.lower() for tag in memory.tags}:
                return self._extract_disease_name(memory)
            if memory.text.startswith("既往歴:"):
                return memory.text.split(":", 1)[1].strip() or None
        return None

    def _extract_disease_name(self, memory: Memory) -> str | None:
        if memory.tags:
            for tag in memory.tags:
                cleaned = tag.strip()
                if cleaned and cleaned != "medical_history":
                    return cleaned
        if memory.text.startswith("既往歴:"):
            return memory.text.split(":", 1)[1].strip() or None
        return None

    def _looks_like_symptom_question(self, lowered_input: str) -> bool:
        return any(keyword in lowered_input for keyword in ["症状", "どんな症状", "痛み", "体調", "具合", "発作", "なにがある"])
