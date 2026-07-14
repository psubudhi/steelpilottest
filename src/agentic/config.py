from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _env_value(*keys: str, default: str | None = None) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return default


def _env_path(*keys: str, default: str) -> Path:
    return Path(_env_value(*keys, default=default) or default)


@dataclass(frozen=True)
class Settings:
    app_name: str = _env_value("STEEL_PILOT_APP_NAME", default="Steel Pilot") or "Steel Pilot"
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    modelling_root: Path = Path(os.getenv("TCM_MODELLING_ROOT", "./tcm_modelling"))
    docs_dir: Path = _env_path("STEEL_PILOT_DOCS_DIR", "STEELCARE_DOCS_DIR", default="./docs")
    vector_dir: Path = _env_path("STEEL_PILOT_VECTOR_DIR", "STEELCARE_VECTOR_DIR", default="./vector_store/faiss_index")
    memory_dir: Path = _env_path("STEEL_PILOT_MEMORY_DIR", "STEELCARE_MEMORY_DIR", default="./memory")
    runtime_dir: Path = _env_path("STEEL_PILOT_RUNTIME_DIR", "STEELCARE_RUNTIME_DIR", default="./data/runtime")
    sqlite_db_path: Path = _env_path("STEEL_PILOT_SQLITE_DB", "STEELCARE_SQLITE_DB", default="./data/runtime/steel_pilot_ops.sqlite")
    langgraph_memory_backend: str = os.getenv("LANGGRAPH_MEMORY_BACKEND", "memory")

    @property
    def model_dir(self) -> Path:
        return self.modelling_root / "models"

    @property
    def output_dir(self) -> Path:
        return self.modelling_root / "outputs"

    @property
    def processed_dir(self) -> Path:
        return self.modelling_root / "data" / "processed"


settings = Settings()
