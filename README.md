# RAG_APP (PDF Ingest + Qdrant + Gemini + Inngest + Streamlit)

Check it out: [Live Link](https://lily010304-rag-app-streamlit-app-5ppc6f.streamlit.app/)

<img width="1889" height="924" alt="upload" src="https://github.com/user-attachments/assets/4f5f2d9d-a456-4005-a60a-f45531807a1b" />

<img width="1222" height="862" alt="question" src="https://github.com/user-attachments/assets/59e4a3b3-1475-4c08-accb-1113a9a85673" />


This repo is a small RAG (Retrieval-Augmented Generation) app:

- **Ingest**: upload a PDF → split into chunks → embed chunks with **Gemini embeddings** → store vectors in **Qdrant**
- **Search/Q&A**: ask a question → embed the question → retrieve top-k chunks from Qdrant → answer using an LLM (via the backend workflow)
- **Orchestration**: **Inngest** runs the ingest + query workflows
- **UI**: **Streamlit** sends events + displays the final answer

## Project layout

- `streamlit_app.py` — Streamlit UI (upload PDF + ask question)
- `main.py` — FastAPI app + Inngest functions (ingest + query)
- `data_loader.py` — PDF loading/chunking + embedding (Gemini)
- `vector_db.py` — Qdrant storage wrapper (upsert + search)
- `custom_types.py` — Pydantic models used by the backend
- `qdrant_storage/` — local Qdrant persistence folder (created/used by Qdrant)

## Prerequisites

1. **Python** (your repo already contains a venv at `rag/`)
2. **Qdrant running locally** (default URL: `http://localhost:6333`)
3. **Gemini API key** (environment variable)
4. **Inngest dev server**

## Environment variables

Create a `.env` file in the repo root (you already have one) and ensure these are set:

- `GEMENI_API` — your Gemini API key
  - Note: the code currently reads `os.environ.get("GEMENI_API")` (spelling matters).
- `INNGEST_API_BASE` (optional) — base URL for the Inngest API used by Streamlit to poll run output
  - Default used by the UI: `http://127.0.0.1:8288/v1`

## Install dependencies

If you are using the existing venv in `rag/`:

PowerShell:

```powershell
.\rag\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

If you don’t have `requirements.txt`, install the key packages manually:

```powershell
python -m pip install streamlit fastapi uvicorn python-dotenv requests qdrant-client llama-index google-genai inngest
```

## Run the system (local dev)

You typically run **three things**:

1) Qdrant
2) Backend (FastAPI + Inngest functions)
3) Inngest dev server
4) Streamlit UI

### 1) Start Qdrant

If you run Qdrant via Docker:

```powershell
docker run --rm -p 6333:6333 -p 6334:6334 -v ${PWD}/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Then verify it’s up:

- Open `http://localhost:6333/` in a browser

### 2) Start the FastAPI backend

From the repo root:

```powershell
.\rag\Scripts\Activate.ps1
python -m uvicorn main:app --reload --port 8000
```

This must match the URL you give to Inngest CLI in the next step.

### 3) Start the Inngest dev server

In another terminal (repo root):

```powershell
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

- `-u .../api/inngest` should point to your FastAPI Inngest handler route.

### 4) Start Streamlit UI

From the repo root:

```powershell
.\rag\Scripts\Activate.ps1
python -m streamlit run .\streamlit_app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`).

## How to use

1. Open the Streamlit page
2. Upload a PDF
3. Wait for “Triggered ingestion …”
4. Ask a question (choose `top_k` chunks)
5. The UI sends an Inngest event and then polls the Inngest API until the run returns output


## Notes

- This is a local-dev setup (Inngest client is created with `is_production=False`).
- If you run multiple ingests, chunks from multiple PDFs may all be stored in the same Qdrant collection (`docs`).
