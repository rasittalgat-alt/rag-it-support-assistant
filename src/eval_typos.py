import json
from pathlib import Path
from typing import List, Dict, Any

from .embeddings_client import EmbeddingsClient
from .vector_db_client import VectorDBClient
from .text_utils import normalize_question


ROOT_DIR = Path(__file__).resolve().parents[1]
TYPOS_PATH = ROOT_DIR / "data" / "eval" / "queries_typos.json"


def load_noisy_queries() -> List[Dict[str, Any]]:
    with TYPOS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def hit_at_k_for_query(
    question: str,
    gold_source_id: str,
    emb_client: EmbeddingsClient,
    vec_client: VectorDBClient,
    k: int,
    use_normalization: bool,
) -> int:
    """
    Считает Hit@k для одного вопроса.
    use_normalization=False  -> baseline (без нормализации)
    use_normalization=True   -> improved (с normalize_question)
    """

    if use_normalization:
        q = normalize_question(question)
    else:
        q = question

    query_vector = emb_client.embed_text(q)
    results = vec_client.search(
        query_vector=query_vector,
        limit=k,
        with_payload=True,
    )

    retrieved_source_ids: List[str] = []
    for hit in results:
        payload = hit.payload or {}
        retrieved_source_ids.append(payload.get("source_id"))

    return 1 if gold_source_id in retrieved_source_ids else 0


def evaluate_typos(k: int = 1) -> None:
    queries = load_noisy_queries()
    print(f"Loaded {len(queries)} noisy queries from {TYPOS_PATH}")

    emb_client = EmbeddingsClient()
    vec_client = VectorDBClient(vector_size=1536)

    baseline_hits = 0
    improved_hits = 0

    for q in queries:
        qid = q["id"]
        question = q["question"]
        gold_source_id = q["gold_source_id"]

        h_base = hit_at_k_for_query(
            question=question,
            gold_source_id=gold_source_id,
            emb_client=emb_client,
            vec_client=vec_client,
            k=k,
            use_normalization=False,
        )

        h_impr = hit_at_k_for_query(
            question=question,
            gold_source_id=gold_source_id,
            emb_client=emb_client,
            vec_client=vec_client,
            k=k,
            use_normalization=True,
        )

        baseline_hits += h_base
        improved_hits += h_impr

        print(f"\nQuery {qid}: {question}")
        print(f"  gold_source_id = {gold_source_id}")
        print(f"  Baseline Hit@{k}: {h_base}")
        print(f"  Improved (normalized) Hit@{k}: {h_impr}")

    total = len(queries)
    base_rate = baseline_hits / total if total > 0 else 0.0
    impr_rate = improved_hits / total if total > 0 else 0.0

    print("\n=== Noisy Retrieval Metrics (Hit@1) ===")
    print(f"Baseline: {baseline_hits}/{total} = {base_rate:.3f}")
    print(f"Improved: {improved_hits}/{total} = {impr_rate:.3f}")

    if base_rate > 0:
        rel_gain = (impr_rate - base_rate) / base_rate * 100
        print(f"Relative improvement: {rel_gain:.1f}%")
    else:
        print("Baseline is 0, relative improvement cannot be computed.")
        

if __name__ == "__main__":
    evaluate_typos(k=1)
