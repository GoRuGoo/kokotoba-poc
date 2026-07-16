from __future__ import annotations

import json
import re
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass


def _extract_text_from_message(message: object) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(str(item.get("text", "")))
            return " ".join(texts)
    return str(message)


class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class MockLLM(LLM):
    def generate(self, prompt: str) -> str:
        if "出力形式:" in prompt:
            return self._generate_memory_json(prompt)

        lines = [line.strip()
                 for line in prompt.strip().splitlines() if line.strip()]
        user_lines = [line for line in lines if line.startswith("user:")]
        question = user_lines[-1].split(":",
                                        1)[1].strip() if user_lines else "ご相談内容"
        return f"{question} について、短く分かりやすく支援します。"

    def _generate_memory_json(self, prompt: str) -> str:
        user_lines = re.findall(r"user:\s*(.+)", prompt)
        assistant_lines = re.findall(r"assistant:\s*(.+)", prompt)
        source_text = user_lines[-1] if user_lines else (
            assistant_lines[-1] if assistant_lines else "")

        memory = source_text.strip() or "会話内容を要約した長期記憶"
        tags = []
        lowered = memory.lower()
        if any(keyword in lowered for keyword in ["病院", "hospital", "受診", "薬"]):
            tags.append("hospital")
        if any(keyword in lowered for keyword in ["保険", "insurance", "マイナ保険証"]):
            tags.append("insurance")
        if not tags:
            tags.append("general")

        payload = {
            "memory": memory,
            "tags": tags,
            "importance": 0.9,
        }
        return json.dumps(payload, ensure_ascii=False)


@dataclass(frozen=True)
class LiteRTLMConfig:
    model_path: str
    model_repo: str = "litert-community/gemma-4-E2B-it-litert-lm"
    model_filename: str = "gemma-4-E2B-it-web.litertlm"
    backend: str = "cpu"


class LiteRTLMGemma4LLM(LLM):
    def __init__(self, config: LiteRTLMConfig) -> None:
        self.config = config
        self._model_path = self._resolve_model_path(config)
        self._litert_lm = self._import_litert_lm()
        self._backend = self._resolve_backend(self._litert_lm, config.backend)
        self._engine = self._litert_lm.Engine(
            str(self._model_path), backend=self._backend
        )
        self._conversation = None

    def _ensure_conversation(self) -> object:
        if self._conversation is not None:
            return self._conversation

        engine = self._engine
        engine.__enter__()
        conversation = engine.create_conversation()
        conversation.__enter__()
        self._conversation = conversation
        return conversation

    def _import_litert_lm(self) -> object:
        try:
            import litert_lm
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "litert_lm is not available. Install LiteRT-LM Python bindings."
            ) from exc
        return litert_lm

    def _resolve_model_path(self, config: LiteRTLMConfig) -> Path:
        model_path = Path(config.model_path)
        if model_path.exists():
            return model_path

        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "huggingface_hub is required to download the Gemma4 LiteRT-LM model"
            ) from exc

        model_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_path = hf_hub_download(
            repo_id=config.model_repo,
            filename=config.model_filename,
            local_dir=str(model_path.parent),
            local_dir_use_symlinks=False,
        )
        return Path(downloaded_path)

    def generate(self, prompt: str) -> str:
        conversation = self._ensure_conversation()
        response = conversation.send_message(
            {
                "role": "user",
                "content": "You are a helpful assistant.\n\n" + prompt,
            }
        )

        return self._extract_response_text(response)

    def _resolve_backend(self, litert_lm: object, backend_name: str) -> object:
        backend = getattr(litert_lm, "Backend", None)
        if backend is None:
            raise RuntimeError("litert_lm.Backend is unavailable")

        backend_key = backend_name.strip().upper()
        if backend_key == "GPU":
            return backend.GPU()
        if backend_key == "NPU":
            return backend.NPU()
        return backend.CPU()

    def _extract_response_text(self, response: object) -> str:
        if isinstance(response, dict):
            content = response.get("content", [])
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(str(item.get("text", "")))
                if texts:
                    return "".join(texts)
        return _extract_text_from_message(response)
