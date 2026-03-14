# Dharmic Wisdom AI – Project Starter

This repository now contains a practical starter blueprint and minimal codebase for building an AI website focused on Dharmic traditions (Hinduism, Jainism, Buddhism, and related Sanskrit/Prakrit/Pali literature).

## 1) Important legal + ethical note first

Before collecting "every book on the internet," you must filter by:

- **Copyright status** (public domain, open license, or explicit permission).
- **Terms of service** of each source website.
- **Religious sensitivity** and contextual accuracy.
- **Transparent citations** for every answer.

Do **not** indiscriminately scrape copyrighted material.

---

## 2) Recommended architecture (RAG-first)

For this domain, use a **Retrieval-Augmented Generation (RAG)** system first, and only fine-tune later if needed.

### Core stack

- **Frontend**: simple web UI (chat + source citations).
- **Backend API**: FastAPI.
- **Vector DB**: Qdrant/Weaviate/pgvector.
- **Embeddings**: multilingual model (supports Sanskrit/Hindi/English).
- **LLM**: API model or self-hosted instruct model.
- **Data pipeline**:
  - source registry
  - downloader
  - OCR/transcription
  - metadata extraction
  - chunking
  - embeddings + indexing

### Why RAG first?

- Faster to launch.
- Easier to update corpus.
- Better source attribution.
- Lower risk than full-domain pretraining.

---

## 3) Suggested phased plan

1. **Corpus policy + source registry**
2. **Ingestion pipeline for legal/open texts**
3. **Search + retrieval quality evaluation**
4. **Answer generation with strict citations**
5. **Safety, doctrinal neutrality, and review tooling**
6. **Optional fine-tuning after strong RAG baseline**

---

## 4) Run the starter

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

### Frontend

```bash
cd frontend
python -m http.server 8080
```

Open `http://localhost:8080`.

---

## 5) What this starter includes

- `backend/app.py`: FastAPI API with `/health` and `/chat` routes.
- `data_pipeline/ingest.py`: extensible ingestion scaffold with source legality gates.
- `frontend/index.html`: basic chat UI that calls backend.

This is a **foundation** you can now extend with real model providers, vector search, OCR, and curated source adapters.
