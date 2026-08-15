import sys

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

DB_DIR = "chroma_db"
COLLECTION = "warden_lore"
MODEL_NAME = "BAAI/bge-base-en-v1.5"

# bge-модели ожидают инструкцию перед поисковым запросом,
# но не перед документами - без неё качество поиска заметно ниже
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEFAULT_QUERIES = [
    "Who leads Aldwyn Hollow?",
    "What is aetheric flux and how is it used?",
    "Which bloodline gift grants the Emberlens?",
]


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(
        path=DB_DIR, settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(COLLECTION)
    print(f"Чанков в коллекции: {collection.count()}\n")

    queries = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEFAULT_QUERIES

    for query in queries:
        vector = model.encode(
            QUERY_PREFIX + query,
            normalize_embeddings=True,
            ).tolist()

        res = collection.query(
            query_embeddings=[vector],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        print("=" * 70)
        print(f"Запрос: {query}\n")
        for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            print(f"[{meta['title']}] чанк {meta['chunk_index'] + 1}"
                  f"/{meta['chunk_total']}, расстояние {dist:.4f}")
            print(f"    {doc[:280].strip()}...\n")


if __name__ == "__main__":
    main()