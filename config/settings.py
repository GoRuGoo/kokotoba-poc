from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    database_path: str = str(Path("data") / "app.sqlite3")
    top_k: int = 5
    embedding_dimension: int = 128
    llm_model_path: str = str(
        Path("data") / "models" / "gemma-4-E2B-it.litertlm")
    llm_model_repo: str = "litert-community/gemma-4-E2B-it-litert-lm"
    llm_model_filename: str = "gemma-4-E2B-it.litertlm"
    llm_backend: str = "cpu"
    llm_use_litert_lm: bool = True


def get_settings() -> AppSettings:
    return AppSettings()
