from typing import List
from openai import OpenAI

from .config import settings


class EmbeddingsClient:
    """
    Обёртка над OpenAI Embeddings (можно заменить на DIAL при необходимости).
    Используется для:
    - генерации вектора для одного текста;
    - батчевой генерации векторов для списка текстов.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.embedding_model

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Please set it in environment variables."
            )

        # Классический клиент OpenAI (новая библиотека)
        self.client = OpenAI(api_key=self.api_key)

    def embed_text(self, text: str) -> List[float]:
        """
        Создаёт embedding для одного текста.
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
        )
        # Берём embedding первого (и единственного) элемента
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Создаёт embeddings для списка текстов.
        """
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]


if __name__ == "__main__":
    """
    Простой тест модуля:
    - убедиться, что ключ работает;
    - посмотреть размерность вектора.
    Перед запуском нужно выставить переменную окружения OPENAI_API_KEY.
    """
    from pprint import pprint

    ec = EmbeddingsClient()
    vec = ec.embed_text("Test embedding for IT Support RAG Assistant.")
    print(f"Vector length: {len(vec)}")
    print("First 5 values:")
    pprint(vec[:5])
