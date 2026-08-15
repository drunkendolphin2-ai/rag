import json
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

KB = Path("knowledge_base")
DB_DIR = "chroma_db"
COLLECTION = "warden_lore"

MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = 1200      # ~200 слов, ~300 токенов
CHUNK_OVERLAP = 200


def load_documents() -> list[tuple[str, str]]:
    docs = []
    for path in sorted(KB.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append((path.name, text))
    return docs


def main() -> None:
    t_start = time.perf_counter()

    docs = load_documents()
    print(f"Документов: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    texts, metadatas, ids = [], [], []
    for filename, content in docs:
        title = Path(filename).stem.replace("_", " ")
        chunks = splitter.split_text(content)
        position = 0
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({
                "source": filename,
                "title": title,
                "chunk_index": i,
                "chunk_total": len(chunks),
                "char_start": position,
                "chars": len(chunk),
            })
            ids.append(f"{Path(filename).stem}::{i}")
            position += len(chunk) - CHUNK_OVERLAP

    print(f"Чанков: {len(texts)}")
    print(f"Средний размер чанка: {sum(len(t) for t in texts) // len(texts)} символов")

    print(f"\nЗагрузка модели {MODEL_NAME}...")
    t_model = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME)
    device = model.device
    print(f"Модель загружена за {time.perf_counter() - t_model:.1f}s, устройство: {device}")
    print(f"Размерность эмбеддингов: {model.get_sentence_embedding_dimension()}")

    print("\nГенерация эмбеддингов...")
    t_embed = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,   # для косинусного расстояния
        convert_to_numpy=True,
    )
    embed_time = time.perf_counter() - t_embed
    print(f"Эмбеддинги готовы за {embed_time:.1f}s "
          f"({len(texts) / embed_time:.1f} чанков/сек)")

    print("\nЗапись в ChromaDB...")
    client = chromadb.PersistentClient(
        path=DB_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    BATCH = 500
    for i in range(0, len(texts), BATCH):
        collection.add(
            ids=ids[i:i + BATCH],
            documents=texts[i:i + BATCH],
            embeddings=embeddings[i:i + BATCH].tolist(),
            metadatas=metadatas[i:i + BATCH],
        )

    total_time = time.perf_counter() - t_start
    print(f"\nВ коллекции: {collection.count()} чанков")
    print(f"Общее время: {total_time:.1f}s")

    stats = {
        "model": MODEL_NAME,
        "dimensions": model.get_sentence_embedding_dimension(),
        "device": str(device),
        "documents": len(docs),
        "chunks": len(texts),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_time_sec": round(embed_time, 1),
        "total_time_sec": round(total_time, 1),
        "collection": COLLECTION,
    }
    Path("index_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Статистика: index_stats.json")


if __name__ == "__main__":
    main()