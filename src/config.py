from dataclasses import dataclass
import os


@dataclass
class Settings:
    """
    Общие настройки проекта.
    Значения можно переопределять через переменные окружения.
    """
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Параметры Qdrant
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name: str = os.getenv("QDRANT_COLLECTION", "it_support_kb")


settings = Settings()
