from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass
class SourceRecord:
    name: str
    url: str
    license: str
    allowed: bool


def is_source_allowed(license_name: str) -> bool:
    allow_list = {"public-domain", "cc-by", "cc-by-sa", "permission-granted"}
    return license_name.strip().lower() in allow_list


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    text = " ".join(text.split())
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def embed_text(base_url: str, model: str, text: str) -> list[float]:
    res = requests.post(
        f"{base_url.rstrip('/')}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=120,
    )
    res.raise_for_status()
    return res.json()["embedding"]


def build_index(corpus_dir: Path, output_path: Path, base_url: str, embed_model: str) -> int:
    rows: list[dict] = []
    for file in sorted(corpus_dir.rglob("*.txt")) + sorted(corpus_dir.rglob("*.md")):
        text = file.read_text(encoding="utf-8", errors="ignore")
        tradition = file.parent.name
        for i, piece in enumerate(chunk_text(text), start=1):
            emb = embed_text(base_url, embed_model, piece)
            rows.append(
                {
                    "title": file.stem,
                    "path": str(file),
                    "tradition": tradition,
                    "chunk_id": i,
                    "text": piece,
                    "embedding": emb,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local RAG index using Ollama embeddings")
    parser.add_argument("--corpus", default="data/corpus", help="Folder containing .txt/.md files")
    parser.add_argument("--out", default="data/index.json", help="Output JSON index")
    parser.add_argument("--ollama", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--embed-model", default="nomic-embed-text", help="Ollama embedding model")
    args = parser.parse_args()

    total = build_index(
        corpus_dir=Path(args.corpus),
        output_path=Path(args.out),
        base_url=args.ollama,
        embed_model=args.embed_model,
    )
    print(f"Indexed {total} chunks into {args.out}")


if __name__ == "__main__":
    main()
