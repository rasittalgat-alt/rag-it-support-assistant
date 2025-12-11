import json
from pathlib import Path
from typing import List, Dict, Any

from .embeddings_client import EmbeddingsClient
from .vector_db_client import VectorDBClient


ROOT_DIR = Path(__file__).resolve().parents[1]
EVAL_QUERIES_PATH = ROOT_DIR / "data" / "eval" / "queries.json"


def load_eval_queries() -> List[Dict[str, Any]]:
    with EVAL_QUERIES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_hits_for_query(
    question: str,
    gold_source_id: str,
    emb_client: EmbeddingsClient,
    vec_client: VectorDBClient,
    top_ks: List[int],
) -> Dict[int, int]:
    """
    Возвращает словарь: {k: 0/1}, попал ли gold_source_id в top-k.
    """
    query_vector = emb_client.embed_text(question)
    max_k = max(top_ks)

    results = vec_client.search(
        query_vector=query_vector,
        limit=max_k,
        with_payload=True,
    )

    retrieved_source_ids: List[str | None] = []
    for hit in results:
        payload = hit.payload or {}
        retrieved_source_ids.append(payload.get("source_id"))

    hits: Dict[int, int] = {}
    for k in top_ks:
        subset = retrieved_source_ids[:k]
        hits[k] = 1 if gold_source_id in subset else 0

    return hits


def evaluate_baseline(top_ks: List[int] = [1, 3, 5]) -> None:
    """
    Базовая оценка (baseline):
    - без фильтров по категории,
    - чистый vector search, как сейчас в RAG.
    """
    queries = load_eval_queries()
    print(f"Loaded {len(queries)} eval queries")

    emb_client = EmbeddingsClient()
    vec_client = VectorDBClient(vector_size=1536)

    # Для каждого k собираем список 0/1
    stats: Dict[int, List[int]] = {k: [] for k in top_ks}

    for q in queries:
        qid = q["id"]
        question = q["question"]
        gold_source_id = q["gold_source_id"]

        hits = compute_hits_for_query(
            question=question,
            gold_source_id=gold_source_id,
            emb_client=emb_client,
            vec_client=vec_client,
            top_ks=top_ks,
        )

        # Можно включить отладочный вывод по каждому запросу
        print(f"\nQuery {qid}: {question}")
        print(f"  gold_source_id = {gold_source_id}")
        for k in top_ks:
            print(f"  Hit@{k}: {hits[k]}")

        for k in top_ks:
            stats[k].append(hits[k])

    print("\n=== Baseline Retrieval Metrics ===")
    total = len(queries)
    for k in top_ks:
        hit_sum = sum(stats[k])
        hit_rate = hit_sum / total if total > 0 else 0.0
        print(f"Hit@{k}: {hit_sum}/{total} = {hit_rate:.3f}")


if __name__ == "__main__":
    evaluate_baseline()
