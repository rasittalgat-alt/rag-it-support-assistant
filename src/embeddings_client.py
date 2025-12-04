from typing import List
from openai import OpenAI

from .config import settings
import os



class EmbeddingsClient:
    def __init__(self):
        # 1. Берём ключ и endpoint из переменных окружения,
        #    которые тебе прислал ментор
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://ai-proxy.lab.epam.com")
        model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small-1")

        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY / OPENAI_API_KEY is not set.")

        # 2. Создаём клиента с кастомным base_url (через прокси)
        self.client = OpenAI(
            api_key=api_key,
            base_url=f"{endpoint}/v1",
        )

        self.model = model

    def embed_text(self, text: str):
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
        )
        return response.data[0].embedding



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
