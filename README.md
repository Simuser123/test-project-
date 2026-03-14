# Dharmic Wisdom AI (Online, OpenAI API)

You asked for an **internet/online setup**, not local Ollama. This project now uses **OpenAI API key**.

## 1) Prerequisites

- Python 3.10+
- OpenAI API key

Set key in shell:

```bash
export OPENAI_API_KEY="your_key_here"
```

## 2) Put your texts in corpus

Add `.txt` or `.md` files in:

- `data/corpus/hinduism/`
- `data/corpus/jainism/`
- `data/corpus/buddhism/`

Use only public-domain/open-license/permitted sources.

## 3) Install dependencies

```bash
cd /workspace/test-project-
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

## 4) Build embedding index (online OpenAI)

```bash
python data_pipeline/ingest.py --corpus data/corpus --out data/index.json
```

Defaults:
- Embedding model: `text-embedding-3-small`
- API URL: `https://api.openai.com/v1`

## 5) Start backend + frontend

Backend:

```bash
cd /workspace/test-project-/backend
uvicorn app:app --reload --port 8000
```

Frontend (new terminal):

```bash
cd /workspace/test-project-/frontend
python -m http.server 8080
```

Open `http://localhost:8080`.

## 6) How this works

1. `data_pipeline/ingest.py` chunks your files and calls OpenAI embeddings.
2. Vectors + text chunks are saved in `data/index.json`.
3. `backend/rag_engine.py` embeds user query, retrieves top chunks by cosine similarity, and calls OpenAI chat completion.
4. UI shows answer with source snippets.

## 7) Suggested next steps

- Move from `data/index.json` to pgvector/Qdrant for scale.
- Add source registry metadata (origin URL, license, translator, edition).
- Add eval set to check citation faithfulness.
- Add moderation and abuse filtering in API layer.
