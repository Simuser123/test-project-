# Dharmic Wisdom AI (Ollama + Python)

You asked: **"I have git, python and Ollama — what to do next?"**

This repo is now a working local starter:
- Ingest your text files into an embedding index using Ollama.
- Ask questions through a FastAPI backend.
- Get answers with source snippets in the frontend.

## 1) Prerequisites

- Python 3.10+
- Ollama installed and running (`ollama serve`)
- Models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.1
```

## 2) Put your texts here

Add `.txt` or `.md` files under:

- `data/corpus/hinduism/`
- `data/corpus/jainism/`
- `data/corpus/buddhism/`

Use only public-domain/open-license/permitted sources.

## 3) Build the local index

```bash
cd /workspace/test-project-
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python data_pipeline/ingest.py --corpus data/corpus --out data/index.json
```

## 4) Start backend + frontend

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

## 5) How retrieval works

1. `data_pipeline/ingest.py` chunks each text file and asks Ollama for embeddings.
2. Chunks + vectors are stored in `data/index.json`.
3. `backend/app.py` embeds your question, retrieves top similar chunks, and sends context to Ollama generation.
4. UI shows answer + cited source snippets.

## 6) Recommended next improvements

- Add a source registry with URL/license metadata per file.
- Add OCR pipeline for scanned PDFs.
- Add evaluation set for factual grounding.
- Add multilingual query normalization (Sanskrit/Hindi/English).
- Replace JSON index with a vector DB (Qdrant/pgvector) for scale.
