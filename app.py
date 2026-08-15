from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from rag import RagBot

bot: RagBot | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot
    bot = RagBot()
    yield


app = FastAPI(title="RAG Knowledge Bot", lifespan=lifespan)


class Query(BaseModel):
    question: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": bot.collection.count()}


@app.post("/ask")
def ask(query: Query) -> dict:
    answer = bot.ask(query.question)
    return {
        "question": answer.question,
        "answer": answer.text,
        "reasoning": answer.reasoning,
        "retrieved": answer.retrieved,
        "sources": [
            {
                "title": s.title,
                "source": s.source,
                "chunk": s.chunk_index,
                "distance": s.distance,
            }
            for s in answer.sources
        ],
    }