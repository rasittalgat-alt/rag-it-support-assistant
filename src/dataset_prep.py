import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any]


def load_faqs() -> List[Document]:
    path = RAW_DIR / "faqs.yaml"
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        faqs = yaml.safe_load(f)

    docs: List[Document] = []
    for item in faqs:
        doc_id = item["id"]
        question = item["question"]
        answer = item["answer"]

        text = f"Question: {question}\n\nAnswer:\n{answer}"
        metadata = {
            "source_type": "faq",
            "source_id": doc_id,
            "category": item.get("category", "other"),
            "title": question,
            "language": "en",
        }
        docs.append(Document(id=doc_id, text=text, metadata=metadata))
    return docs


def load_tickets() -> List[Document]:
    path = RAW_DIR / "tickets.json"
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        tickets = json.load(f)

    docs: List[Document] = []
    for item in tickets:
        doc_id = item["ticket_id"]
        title = item.get("title", "")
        description = item.get("description", "")
        resolution = item.get("resolution", "")

        text = f"Ticket ID: {doc_id}\nTitle: {title}\n\nDescription:\n{description}\n\nResolution:\n{resolution}"
        metadata = {
            "source_type": "ticket",
            "source_id": doc_id,
            "category": item.get("category", "other"),
            "title": title,
            "language": "en",
        }
        docs.append(Document(id=doc_id, text=text, metadata=metadata))
    return docs


def load_markdown_dir(subdir: str, source_type: str) -> List[Document]:
    """
    Загружает все .md файлы из data/raw/<subdir>
    source_type: 'runbook' или 'policy'
    """
    base_dir = RAW_DIR / subdir
    if not base_dir.exists():
        return []

    docs: List[Document] = []

    for path in base_dir.glob("*.md"):
        with path.open("r", encoding="utf-8") as f:
            content = f.read()

        # Заголовок берём из первой строки с "# ", если есть
        title = path.stem
        for line in content.splitlines():
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break

        # Категорию грубо берём из имени файла (wifi, vpn, password, sla и т.п.)
        category = path.stem.split("_")[0].lower()

        doc_id = f"{source_type}_{path.stem}"

        metadata = {
            "source_type": source_type,
            "source_id": doc_id,
            "category": category,
            "title": title,
            "language": "en",
            "filename": path.name,
        }

        docs.append(Document(id=doc_id, text=content, metadata=metadata))

    return docs


def simple_chunk_text(text: str, max_chars: int = 700, overlap: int = 100) -> List[str]:
    """
    Очень простой чанкер: режет текст на куски по max_chars
    с перекрытием overlap символов.
    Для учебного проекта достаточно.
    """
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == length:
            break

        start = end - overlap  # немного перекрываем контекст

    return chunks


def build_chunks() -> None:
    """
    Собирает все документы, режет на чанки и сохраняет в chunks.jsonl
    """
    docs: List[Document] = []
    docs.extend(load_faqs())
    docs.extend(load_tickets())
    docs.extend(load_markdown_dir("runbooks", "runbook"))
    docs.extend(load_markdown_dir("policies", "policy"))

    print(f"Loaded documents: {len(docs)}")

    with CHUNKS_PATH.open("w", encoding="utf-8") as out_f:
        count_chunks = 0
        for doc in docs:
            chunks = simple_chunk_text(doc.text, max_chars=700, overlap=100)
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{doc.id}_chunk_{idx:03d}"

                record = {
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": doc.metadata,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count_chunks += 1

    print(f"Saved {count_chunks} chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    build_chunks()
