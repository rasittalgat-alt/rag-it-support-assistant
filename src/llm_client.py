from typing import List, Dict, Any
import os

from openai import OpenAI


class LLMClient:
    """
    Клиент LLM, поддерживающий:
    - прямой доступ к OpenAI (через OPENAI_API_KEY),
    - либо EPAM ai-proxy (через AZURE_OPENAI_API_KEY, AZURE_OPENAI_*).

    Основной метод:
    - generate_answer(question, context_chunks) -> str
    """

    def __init__(self) -> None:
        openai_key = os.getenv("OPENAI_API_KEY")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")

        # ----- Режим 1: прямой OpenAI -----
        if openai_key:
            self.client = OpenAI(api_key=openai_key)
            # можно поменять на gpt-4o, если доступен
            self.model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
            print(f"[LLMClient] Using direct OpenAI ({self.model}).")

        # ----- Режим 2: EPAM ai-proxy / Azure совместимый -----
        elif azure_key:
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://ai-proxy.lab.epam.com")
            deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini-1")

            self.client = OpenAI(
                api_key=azure_key,
                base_url=f"{endpoint}/v1",
            )
            self.model = deployment
            print(f"[LLMClient] Using AI proxy: {endpoint}, deployment={deployment}")

        else:
            raise ValueError(
                "Neither OPENAI_API_KEY nor AZURE_OPENAI_API_KEY is set. "
                "Set OPENAI_API_KEY for direct OpenAI or AZURE_OPENAI_API_KEY for ai-proxy."
            )

    def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        """
        Генерирует ответ на вопрос, используя переданные чанки как контекст.

        context_chunks: список словарей вида:
        {
          "text": "...",
          "metadata": {...},
          "score": 0.98
        }
        """

        # Собираем текст контекста
        context_parts = []
        for idx, item in enumerate(context_chunks, start=1):
            context_parts.append(
                f"[Doc {idx} | score={item.get('score'):.3f}]\n{item['text']}"
            )
        context_text = "\n\n".join(context_parts)

        system_prompt = (
            "You are an IT Support assistant. "
            "Use the provided context as the main source of truth to answer the question. "
            "The user question may contain typos or be more general than the examples in the context. "
            "If the context is related to the question, use it and generalize from it to provide "
            "the best possible practical answer. "
            "Only if the context is clearly unrelated to the question, say that you don't know "
            "and suggest contacting IT Support."
        )


        user_content = (
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION:\n{question}\n\n"
            "Answer in a clear, concise way. "
            "If there are several possible solutions, list them as steps."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )

        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # Мини-тест: без RAG, просто проверяем, что LLM работает.
    client = LLMClient()
    answer = client.generate_answer(
        question="How can I reset my corporate password?",
        context_chunks=[
            {
                "text": "To reset your corporate password you should open the portal "
                        "https://password.company.com, confirm your identity and "
                        "set a new password that follows the password policy.",
                "metadata": {},
                "score": 1.0,
            }
        ],
    )
    print("LLM answer:\n", answer)
