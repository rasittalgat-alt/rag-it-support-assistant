from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .config import settings


class VectorDBClient:
    """
    Обёртка над Qdrant:
    - создание коллекции;
    - добавление точек;
    - поиск.
    """

    def __init__(self,
                 host: str | None = None,
                 port: int | None = None,
                 collection_name: str | None = None,
                 vector_size: int = 1536,
                 distance: qm.Distance = qm.Distance.COSINE):
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = collection_name or settings.collection_name
        self.vector_size = vector_size
        self.distance = distance

        self.client = QdrantClient(host=self.host, port=self.port)

    def create_collection_if_not_exists(self) -> None:
        collections = self.client.get_collections().collections
        existing_names = {c.name for c in collections}

        if self.collection_name in existing_names:
            print(f"Collection '{self.collection_name}' already exists")
            return

        print(f"Creating collection '{self.collection_name}'...")
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=qm.VectorParams(
                size=self.vector_size,
                distance=self.distance,
            ),
        )
        print("Collection created.")

    def upsert_points(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> None:
        """
        Добавляет или обновляет точки в коллекции.
        """
        self.client.upsert(
            collection_name=self.collection_name,
            points=qm.Batch(
                ids=ids,
                vectors=vectors,
                payloads=payloads,
            ),
        )

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        with_payload: bool = True,
    ):
        """
        Выполняет поиск ближайших векторов.

        Для qdrant-client 1.x используем метод query_points:
        - collection_name: имя коллекции
        - query: сам вектор запроса
        - with_payload: вернуть payload
        - limit: сколько результатов
        """

        res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            with_payload=with_payload,
            limit=limit,
        )

        # В res лежит объект с полем .points (список ScoredPoint)
        return res.points

    def search_with_category(
        self,
        query_vector: List[float],
        category: str | None,
        limit: int = 5,
        with_payload: bool = True,
    ):
        """
        Поиск с фильтром по категории (payload['category']).

        Если category = None или пустая строка — просто fallback на обычный search().
        """

        if not category:
            return self.search(
                query_vector=query_vector,
                limit=limit,
                with_payload=with_payload,
            )

        qfilter = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="category",
                    match=qm.MatchValue(value=category),
                )
            ]
        )

        res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=qfilter,
            with_payload=with_payload,
            limit=limit,
        )

        return res.points



