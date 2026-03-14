from __future__ import annotations

import json
import os
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
        openai_base_url: str = "https://api.openai.com/v1",
        embedding_model: str = "text-embedding-3-small",
        generation_model: str = "gpt-4o-mini",
    ) -> None:
        self.index_path = Path(index_path)
        self.openai_base_url = openai_base_url.rstrip("/")
        self.embedding_model = embedding_model
        self.generation_model = generation_model

    def load_index(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    @staticmethod
    def _api_key() -> str:
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return key

    def _embed(self, text: str) -> np.ndarray:
        res = requests.post(
            f"{self.openai_base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
            },
            json={"model": self.embedding_model, "input": text},
            timeout=120,
        )
        res.raise_for_status()
        data = res.json()
        return np.array(data["data"][0]["embedding"], dtype=np.float32)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
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
            return "I could not find indexed passages. Run the ingestion step first, then ask again."

        context_blocks = []
        for i, chunk in enumerate(retrieved, start=1):
            context_blocks.append(f"[Source {i}] title={chunk.title} path={chunk.path}\n{chunk.text}")

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
            f"{self.openai_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.generation_model,
                "messages": [
                    {"role": "system", "content": "Give accurate answers from supplied sources only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            },
            timeout=240,
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()
