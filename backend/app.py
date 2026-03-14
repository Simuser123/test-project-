from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import requests

from rag_engine import RagEngine


app = FastAPI(title="Dharmic Wisdom AI API", version="0.2.0")
engine = RagEngine()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User question")
    tradition: str | None = Field(
        default=None,
        description="Optional filter, e.g. hinduism, jainism, buddhism",
    )


class Source(BaseModel):
    title: str
    url: str
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict[str, str | int]:
    total = len(engine.load_index())
    return {"status": "ok", "indexed_chunks": total}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        retrieved = engine.retrieve(req.question, tradition=req.tradition, top_k=4)
        answer = engine.answer(req.question, retrieved, tradition=req.tradition)
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not reach Ollama. Ensure Ollama is running and models are pulled "
                "(nomic-embed-text, llama3.1)."
            ),
        ) from exc

    sources = [
        Source(
            title=chunk.title,
            url=chunk.path,
            excerpt=chunk.text[:300],
        )
        for chunk in retrieved
    ]

    return ChatResponse(answer=answer, sources=sources)
