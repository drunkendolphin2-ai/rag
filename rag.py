from dataclasses import dataclass, field

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from sentence_transformers import SentenceTransformer

import config
import defense
from prompts import (
    FEW_SHOT,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_UNSAFE,
    build_user_message,
)


@dataclass
class Source:
    title: str
    source: str
    chunk_index: int
    distance: float


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Source] = field(default_factory=list)
    retrieved: bool = True
    reasoning: str = ""
    blocked_chunks: list[str] = field(default_factory=list)


NO_ANSWER = "I don't know."


class RagBot:
    def __init__(self) -> None:
        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        client = chromadb.PersistentClient(
            path=config.DB_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_collection(config.COLLECTION)
        self.llm = OpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            timeout=120.0,
        )

    # ---------- retrieval ----------

    def retrieve(self, question: str) -> list[tuple[str, Source]]:
        vector = self.embedder.encode(
            config.QUERY_PREFIX + question,
            normalize_embeddings=True,
            ).tolist()

        res = self.collection.query(
            query_embeddings=[vector],
            n_results=config.TOP_K,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            # слой 1: нерелевантное отсекается до вызова LLM
            if dist > config.DISTANCE_THRESHOLD:
                continue
            hits.append((doc, Source(
                title=meta["title"],
                source=meta["source"],
                chunk_index=meta["chunk_index"],
                distance=round(dist, 4),
            )))
        return hits

    # ---------- защита от промпт-инъекций ----------

    def filter_chunks(
            self, hits: list[tuple[str, Source]]
    ) -> tuple[list[tuple[str, Source]], list[str]]:
        """Слои 2 и 3: санитизация и отбрасывание подозрительных чанков."""
        kept: list[tuple[str, Source]] = []
        blocked: list[str] = []

        for doc, src in hits:
            found = defense.detect(doc)

            if not found:
                kept.append((doc, src))
                continue

            # слой 3: чанк отбрасывается целиком
            if config.DEFENSE_POSTCHECK:
                blocked.append(f"{src.title}: отброшен, найдено {found}")
                continue

            # слой 2: управляющие конструкции вырезаются, текст остаётся
            if config.DEFENSE_SANITIZE:
                doc = defense.sanitize(doc)
                blocked.append(f"{src.title}: санитизирован, найдено {found}")

            kept.append((doc, src))

        return kept, blocked

    # ---------- generation ----------

    def ask(self, question: str) -> Answer:
        hits = self.retrieve(question)

        if not hits:
            return Answer(
                question=question,
                text=NO_ANSWER,
                retrieved=False,
            )

        kept, blocked = self.filter_chunks(hits)

        if not kept:
            return Answer(
                question=question,
                text=NO_ANSWER,
                retrieved=False,
                blocked_chunks=blocked,
            )

        context = "\n\n".join(f"[{src.title}] {doc}" for doc, src in kept)

        # слой 1 защиты промпта: правило "контекст - это данные, а не команды"
        system = SYSTEM_PROMPT if config.DEFENSE_PREPROMPT else SYSTEM_PROMPT_UNSAFE

        messages = [{"role": "system", "content": system}]
        messages += FEW_SHOT
        messages.append({
            "role": "user",
            "content": build_user_message(context, question),
        })

        completion = self.llm.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            temperature=0.0,
        )
        raw = (completion.choices[0].message.content or "").strip()

        reasoning, text = "", raw
        if "Answer:" in raw:
            head, _, tail = raw.partition("Answer:")
            reasoning = head.replace("Reasoning:", "").strip()
            text = tail.strip()

        return Answer(
            question=question,
            text=text,
            sources=[src for _, src in kept],
            reasoning=reasoning,
            blocked_chunks=blocked,
        )