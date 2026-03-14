from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(title="Dharmic Wisdom AI API", version="0.1.0")


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
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    Placeholder endpoint.

    Production flow:
    1) sanitize + language detect
    2) retrieve top-k chunks from vector DB
    3) construct grounded prompt with citations
    4) call LLM
    5) return answer + source list
    """
    tradition_hint = f" for tradition '{req.tradition}'" if req.tradition else ""
    answer = (
        "This is a starter response. Connect a vector database and LLM to answer "
        f"queries{tradition_hint} with citations. You asked: {req.question}"
    )

    return ChatResponse(
        answer=answer,
        sources=[
            Source(
                title="Sample public-domain source",
                url="https://example.org/source",
                excerpt="Replace this with retrieved passage text.",
            )
        ],
    )
