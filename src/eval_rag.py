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


def classify_category(question: str) -> str | None:
    """
    Очень простой rule-based классификатор категории по тексту вопроса.
    Категории должны совпадать с тем, что хранится в payload['category'].
    """
    q = question.lower()

    # Wi-Fi / сеть
    if "wifi" in q or "wi-fi" in q or "wireless" in q:
        return "wifi"

    # VPN
    if "vpn" in q or "anyconnect" in q:
        return "vpn"

    # Email / Outlook
    if "outlook" in q or "email" in q or "mail" in q or "webmail" in q:
        return "email"

    # Принтеры
    if "printer" in q or "print " in q or "print job" in q:
        return "printer"

    # SLA / политики
    if "sla" in q or "priority" in q or "p1" in q or "critical incident" in q:
        return "it"

    # Политика паролей
    if "password" in q and (
        "complexity" in q
        or "requirement" in q
        or "requirements" in q
        or "rules" in q
        or "policy" in q
    ):
        return "password"

    # Аккаунт / обычный password reset
    if "password" in q or "account" in q or "login" in q:
        return "account"

    return None



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


def compute_hits_for_query_with_category(
    question: str,
    gold_source_id: str,
    emb_client: EmbeddingsClient,
    vec_client: VectorDBClient,
    top_ks: List[int],
) -> Dict[int, int]:
    """
    То же самое, что compute_hits_for_query, но:
    - сначала определяем категорию вопроса,
    - если нашли category, делаем search_with_category,
    - иначе fallback на обычный search().
    """
    query_vector = emb_client.embed_text(question)
    max_k = max(top_ks)

    category = classify_category(question)

    if category:
        results = vec_client.search_with_category(
            query_vector=query_vector,
            category=category,
            limit=max_k,
            with_payload=True,
        )
    else:
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



def evaluate(top_ks: List[int] = [1, 3, 5]) -> None:
    """
    Сравнение:
    - baseline retrieval (без фильтров);
    - category-aware retrieval (с фильтром по категории, если она распознана).
    """
    queries = load_eval_queries()
    print(f"Loaded {len(queries)} eval queries")

    emb_client = EmbeddingsClient()
    vec_client = VectorDBClient(vector_size=1536)

    # Для каждого k собираем список 0/1
    stats_baseline: Dict[int, List[int]] = {k: [] for k in top_ks}
    stats_category: Dict[int, List[int]] = {k: [] for k in top_ks}

    for q in queries:
        qid = q["id"]
        question = q["question"]
        gold_source_id = q["gold_source_id"]

        baseline_hits = compute_hits_for_query(
            question=question,
            gold_source_id=gold_source_id,
            emb_client=emb_client,
            vec_client=vec_client,
            top_ks=top_ks,
        )

        category_hits = compute_hits_for_query_with_category(
            question=question,
            gold_source_id=gold_source_id,
            emb_client=emb_client,
            vec_client=vec_client,
            top_ks=top_ks,
        )

        print(f"\nQuery {qid}: {question}")
        print(f"  gold_source_id = {gold_source_id}")
        print(f"  predicted_category = {classify_category(question)}")
        for k in top_ks:
            print(f"  Baseline Hit@{k}: {baseline_hits[k]}")
        for k in top_ks:
            print(f"  Category-aware Hit@{k}: {category_hits[k]}")

        for k in top_ks:
            stats_baseline[k].append(baseline_hits[k])
            stats_category[k].append(category_hits[k])

    total = len(queries)

    print("\n=== Retrieval Metrics (Baseline) ===")
    for k in top_ks:
        hit_sum = sum(stats_baseline[k])
        hit_rate = hit_sum / total if total > 0 else 0.0
        print(f"Hit@{k}: {hit_sum}/{total} = {hit_rate:.3f}")

    print("\n=== Retrieval Metrics (Category-aware) ===")
    for k in top_ks:
        hit_sum = sum(stats_category[k])
        hit_rate = hit_sum / total if total > 0 else 0.0
        print(f"Hit@{k}: {hit_sum}/{total} = {hit_rate:.3f}")



if __name__ == "__main__":
    evaluate()
