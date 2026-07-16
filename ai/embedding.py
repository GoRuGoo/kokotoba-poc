from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[float]:
        raise NotImplementedError


class HashEmbeddingModel(EmbeddingModel):
    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = text.lower().split()

        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(self.dimension):
                vector[index] += digest[index % len(digest)] / 255.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector

        return [value / norm for value in vector]


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "sentence-transformers is not available") from exc

        self._model = SentenceTransformer(model_name)

    def encode(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()


class EmbeddingService:
    def __init__(self, model: EmbeddingModel | None = None, dimension: int = 128) -> None:
        self._model = model or HashEmbeddingModel(dimension=dimension)

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text)
