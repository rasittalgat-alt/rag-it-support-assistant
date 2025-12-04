from typing import List
import os

from openai import OpenAI
# from .config import settings  # если не используется, можно оставить закомментированным


class EmbeddingsClient:
    def __init__(self) -> None:
        """
        Клиент эмбеддингов, который умеет работать:
        - либо напрямую с OpenAI (через OPENAI_API_KEY),
        - либо через EPAM ai-proxy (через AZURE_OPENAI_*).
        """

        openai_key = os.getenv("OPENAI_API_KEY")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")

        if openai_key:
            # Вариант 1: обычный OpenAI
            self.client = OpenAI(api_key=openai_key)
            # стандартная модель эмбеддингов
            self.model = "text-embedding-3-small"
            print("[EmbeddingsClient] Using direct OpenAI (text-embedding-3-small).")
        elif azure_key:
            # Вариант 2: EPAM ai-proxy
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://ai-proxy.lab.epam.com")
            deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small-1")

            self.client = OpenAI(
                api_key=azure_key,
                base_url=f"{endpoint}/v1",
            )
            self.model = deployment
            print(f"[EmbeddingsClient] Using AI proxy: {endpoint}, deployment={deployment}")
        else:
            raise ValueError(
                "Neither OPENAI_API_KEY nor AZURE_OPENAI_API_KEY is set. "
                "Set OPENAI_API_KEY for direct OpenAI or AZURE_OPENAI_API_KEY for ai-proxy."
            )

    def embed_text(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
        )
        return response.data[0].embedding


if __name__ == "__main__":
    ec = EmbeddingsClient()
    vec = ec.embed_text("Test embedding for IT Support RAG Assistant.")
    print(f"Vector length: {len(vec)}")
    print("First 5 values:")
    print(vec[:5])
