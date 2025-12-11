from typing import List, Dict, Any

from .embeddings_client import EmbeddingsClient
from .vector_db_client import VectorDBClient
from .llm_client import LLMClient
from .text_utils import normalize_question


class RAGPipeline:
    """
    Основной RAG-пайплайн:
    - embed запроса,
    - поиск по Qdrant,
    - генерация ответа с использованием контекста.
    """

    def __init__(self, top_k: int = 5):
        self.emb_client = EmbeddingsClient()
        self.vec_client = VectorDBClient(vector_size=1536)
        self.llm_client = LLMClient()
        self.top_k = top_k

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        """
        Возвращает top_k чанков из Qdrant в формате:
        [
            {
                "text": "...",
                "metadata": {...},
                "score": 0.98,
            },
            ...
        ]
        """

        # 1. Нормализуем вопрос (исправляем частые опечатки)
        normalized_question = normalize_question(question)

        # 2. Делаем эмбеддинг уже нормализованного текста
        query_vector = self.emb_client.embed_text(normalized_question)

        # 3. Ищем похожие вектора в Qdrant
        results = self.vec_client.search(
            query_vector=query_vector,
            limit=self.top_k,
        )

        # 4. Приводим результаты к удобному формату
        docs: List[Dict[str, Any]] = []
        for hit in results:
            payload = hit.payload or {}
            text = payload.get("text", "")
            docs.append(
                {
                    "text": text,
                    "metadata": payload,
                    "score": hit.score,
                }
            )

        return docs

    def answer_question(self, question: str, temperature: float = 0.1) -> Dict[str, Any]:
        """
        Полный цикл RAG:
        - нормализуем вопрос (исправляем частые опечатки),
        - ищем top_k релевантных чанков,
        - отправляем нормализованный вопрос и список чанков в LLM,
        - возвращаем ответ + использованный контекст.
        """

        # 1. Нормализуем вопрос (для поиска и для LLM)
        normalized_question = normalize_question(question)

        # 2. Получаем документы из Qdrant (retrieve уже собирает их в нужный формат)
        docs: List[Dict[str, Any]] = self.retrieve(question)

        # 3. Генерируем ответ, используя НОРМАЛИЗОВАННЫЙ вопрос и чанки как контекст
        answer = self.llm_client.generate_answer(
            question=normalized_question,
            context_chunks=docs,
            temperature=temperature,
        )

        # 4. Возвращаем всё, что нужно UI
        return {
            "answer": answer,
            "question": question,
            "normalized_question": normalized_question,
            "docs": docs,
        }



if __name__ == "__main__":
    pipeline = RAGPipeline(top_k=4)
    user_question = "How can I connect to corporate Wi-Fi on Windows?"

    result = pipeline.answer_question(user_question)

    print("QUESTION:")
    print(result["question"])
    print("\nANSWER:")
    print(result["answer"])
    print("\n--- CONTEXT DOCS ---")
    for i, doc in enumerate(result["documents"], start=1):
        print(f"\n[Doc {i} | score={doc['score']:.3f}]")
        print(doc["text"][:300], "...")
