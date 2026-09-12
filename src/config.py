<<<<<<< HEAD
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings. Override via environment or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "G-RAG"
    VERSION: str = "2.1.0"

    VECTOR_DB_PATH: str = "vector_db"
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "phi3:mini"

    # Document chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Retrieval / pipeline
    TOP_K_RETRIEVAL: int = 10
    TOP_K_FINAL: int = 5
    COMPRESSION_THRESHOLD: float = 0.3
    MEMORY_TURNS: int = 5
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    EMBED_MODEL: str = "all-MiniLM-L6-v2"

    # Optional: expose evaluation scores in API/UI (still computed when True)
    RUN_EVALUATION: bool = True
    DEBUG_MODE: bool = False

    # Kept optional for deploy secrets compatibility; unused by local G-RAG
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None


@lru_cache()
def get_settings() -> Settings:
    return Settings()
=======
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings. Override via environment or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "G-RAG AI Assistant"
    VERSION: str = "2.1.0"

    VECTOR_DB_PATH: str = "vector_db"
    OLLAMA_HOST: str = "http://localhost:11434"
    # Default is a small model suitable for low-RAM machines.
    # llama3.2:1b follows instructions better than tinyllama.
    # phi3:mini needs ~5 GiB free memory.
    OLLAMA_MODEL: str = "llama3.2:1b"
    # Keep the model loaded in Ollama between requests (e.g. "10m", "0" = unload)
    OLLAMA_KEEP_ALIVE: str = "10m"
    # Cap answer length for faster generation
    LLM_NUM_PREDICT: int = 192

    # Document chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Retrieval / pipeline (smaller = faster)
    TOP_K_RETRIEVAL: int = 6
    TOP_K_FINAL: int = 3
    COMPRESSION_THRESHOLD: float = 0.3
    MEMORY_TURNS: int = 5
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    EMBED_MODEL: str = "all-MiniLM-L6-v2"

    # Speed toggles — heavy steps off by default on low-RAM machines
    ENABLE_QUERY_REWRITE: bool = False
    ENABLE_RERANK: bool = False
    ENABLE_COMPRESSION: bool = False

    # Extra Ollama calls for faithfulness/relevance scores (disable on low RAM)
    RUN_EVALUATION: bool = False
    DEBUG_MODE: bool = False

    # Safety PIN for "Reset" (delete all documents)
    RESET_DOCUMENTS_PIN: str = "1234"

    # Kept optional for deploy secrets compatibility; unused by local G-RAG
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None


@lru_cache()
def get_settings() -> Settings:
    return Settings()
>>>>>>> 0975fe4 (feat: speed up RAG pipeline and add PIN-protected document reset)
