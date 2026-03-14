from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests


@dataclass
class RetrievedChunk:
    title: str
    path: str
    text: str
    score: float


class RagEngine:
    def __init__(
        self,
        index_path: str = "data/index.json",
        ollama_base_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        generation_model: str = "llama3.1",
    ) -> None:
        self.index_path = Path(index_path)
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.embedding_model = embedding_model
        self.generation_model = generation_model

    def load_index(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _embed(self, text: str) -> np.ndarray:
        res = requests.post(
            f"{self.ollama_base_url}/api/embeddings",
            json={"model": self.embedding_model, "prompt": text},
            timeout=120,
        )
        res.raise_for_status()
        data = res.json()
        return np.array(data["embedding"], dtype=np.float32)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def retrieve(self, question: str, tradition: str | None = None, top_k: int = 4) -> list[RetrievedChunk]:
        rows = self.load_index()
        if tradition:
            rows = [r for r in rows if r.get("tradition", "").lower() == tradition.lower()]
        if not rows:
            return []

        q_emb = self._embed(question)
        scored: list[RetrievedChunk] = []

        for row in rows:
            emb = np.array(row["embedding"], dtype=np.float32)
            score = self._cosine_similarity(q_emb, emb)
            scored.append(
                RetrievedChunk(
                    title=row["title"],
                    path=row["path"],
                    text=row["text"],
                    score=score,
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def answer(self, question: str, retrieved: list[RetrievedChunk], tradition: str | None = None) -> str:
        if not retrieved:
            return (
                "I could not find indexed passages. Run the ingestion step first, then ask again."
            )

        context_blocks = []
        for i, chunk in enumerate(retrieved, start=1):
            context_blocks.append(
                f"[Source {i}] title={chunk.title} path={chunk.path}\n{chunk.text}"
            )

        tradition_line = f"Tradition filter: {tradition}\n" if tradition else ""
        prompt = (
            "You are a careful assistant for Dharmic texts. "
            "Answer only from provided context and cite sources like [Source 1].\n"
            f"{tradition_line}"
            "Context:\n"
            + "\n\n".join(context_blocks)
            + f"\n\nQuestion: {question}\nAnswer:"
        )

        res = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={"model": self.generation_model, "prompt": prompt, "stream": False},
            timeout=240,
        )
        res.raise_for_status()
        return res.json().get("response", "")
