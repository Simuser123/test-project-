# Dharmic Wisdom AI (Online, OpenAI API)

This project runs online using OpenAI-compatible APIs.

## 1) Create `.env` in project root

Create `/workspace/test-project-/.env` and paste:

```env
OPENAI_API_KEY=sk-qrst1234qrst1234qrst1234qrst1234qrst1234
OPENAI_BASE_URL=https://models.inference.ai.azure.com
```

> `OPENAI_BASE_URL` can point to OpenAI-compatible endpoints (including Azure-hosted compatible gateways).

## 2) Install dependencies

```bash
cd /workspace/test-project-
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

## 3) Put your text files in corpus

- `data/corpus/hinduism/`
- `data/corpus/jainism/`
- `data/corpus/buddhism/`

Use only public-domain/open-license/permitted sources.

## 4) Build embedding index

```bash
python data_pipeline/ingest.py --corpus data/corpus --out data/index.json
```

This command now auto-reads `OPENAI_API_KEY` and `OPENAI_BASE_URL` from `.env`.

## 5) Run backend + frontend

Backend:

```bash
cd /workspace/test-project-/backend
uvicorn app:app --reload --port 8000
```

Frontend:

```bash
cd /workspace/test-project-/frontend
python -m http.server 8080
```

Open `http://localhost:8080`.

## 6) Notes

- Backend and ingestion both load `.env` automatically.
- If your provider has different model IDs, pass them in code/CLI as needed.
