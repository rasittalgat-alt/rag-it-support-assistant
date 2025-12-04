import json
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm

from .config import settings
from .embeddings_client import EmbeddingsClient
from .vector_db_client import VectorDBClient


ROOT_DIR = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT_DIR / "data" / "processed" / "chunks.jsonl"


def load_chunks() -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def ingest(batch_size: int = 16) -> None:
    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Total chunks: {len(chunks)}")

    emb_client = EmbeddingsClient()
    vec_client = VectorDBClient(vector_size=1536)

    # создаём коллекцию, если её ещё нет
    vec_client.create_collection_if_not_exists()

    # батчами создаём вектора и отправляем в Qdrant
    for i in tqdm(range(0, len(chunks), batch_size), desc="Ingesting"):
        batch = chunks[i : i + batch_size]

        ids = [item["id"] for item in batch]
        texts = [item["text"] for item in batch]
        payloads = [item["metadata"] | {"text": item["text"]} for item in batch]

        vectors = emb_client.embed_batch(texts)
        vec_client.upsert_points(ids=ids, vectors=vectors, payloads=payloads)

    print("Ingestion completed.")


if __name__ == "__main__":
    ingest()
