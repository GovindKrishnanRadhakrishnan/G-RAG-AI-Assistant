# G-RAG

Local document Q&A powered by retrieval-augmented generation (RAG) and **Ollama**.

Upload PDFs, ask questions in plain language, and get answers with readable source citations — all running on your machine.

---

## Features

- PDF upload and processing
- Hybrid retrieval (keyword + semantic search)
- Re-ranking and context compression
- Local LLM answers via Ollama
- Source citations (document name + page)
- Conversation memory across turns
- Single HTML + FastAPI UI

---

## Architecture

```text
Browser (templates/index.html)
        │
        ▼
FastAPI (main.py)  ──►  Document processing  ──►  Embeddings (sentence-transformers)
        │                                              │
        │                                              ▼
        │                                         ChromaDB (vector_db/)
        │                                              │
        └──────── RAG pipeline (src/rag/) ◄────────────┘
                        │
                        ▼
                   Ollama (local LLM)
```

Optional cloud/K8s/CI assets live under `deploy/` and are **not** required for local use.

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running
- A chat model pulled locally (default: `phi3:mini`)

---

## Installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

---

## Ollama Setup

1. Install and start Ollama.
2. Pull the model configured in `.env` (default `phi3:mini`):

```bash
ollama pull phi3:mini
```

3. Confirm Ollama is reachable at `http://localhost:11434`.

---

## Configuration

Copy `.env.example` to `.env`. Important settings:

| Variable | Purpose | Default |
|----------|---------|---------|
| `OLLAMA_HOST` | Ollama base URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Chat / rewrite / eval model | `phi3:mini` |
| `VECTOR_DB_PATH` | Chroma persistence path | `vector_db` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | PDF chunking | `1000` / `200` |
| `TOP_K_RETRIEVAL` / `TOP_K_FINAL` | Retrieval depth | `10` / `5` |

Do not commit `.env`.

---

## Database Setup

G-RAG uses **ChromaDB** on disk under `vector_db/`.

Existing data is preserved. The collection name remains `research_papers` for compatibility — do not rename it until a planned migration.

---

## Running the Application

```bash
uvicorn main:app --host 127.0.0.1 --port 8001
```

Open: [http://127.0.0.1:8001](http://127.0.0.1:8001)

### Docker Compose (optional)

```bash
docker compose up --build
docker exec -it grag-ollama ollama run phi3:mini
```

App: [http://localhost:8001](http://localhost:8001)

---

## Document Ingestion

1. Open the app in your browser.
2. Upload one or more PDFs in the sidebar.
3. Click **Process document**.
4. Wait until the status shows the document is ready.
5. Ask questions in the chat box.

---

## Using the RAG

- Ask natural-language questions about uploaded documents.
- Expand **Sources** under an answer to see file name, page, and a short snippet.
- Use **Clear conversation** to reset chat memory.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | App, Ollama, and vector store health |
| `POST` | `/api/v1/upload-paper` | Upload and process a PDF |
| `POST` | `/api/v1/chat` | Ask a question (`{"question": "..."}`) |
| `POST` | `/api/v1/clear` | Clear conversation memory |
| `GET` | `/metrics` | Prometheus metrics (if installed) |

---

## Testing

```bash
pytest -q
```

Tests mock Ollama and embedding models so they can run offline.

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| Ollama unreachable | Start Ollama; check `OLLAMA_HOST`; run `ollama list` |
| Empty / weak answers | Upload a PDF first; confirm processing succeeded |
| Upload fails | Use a valid PDF; check server logs |
| Slow first query | Embedding / cross-encoder models download once on first use |

---

## Project Structure

```text
G-RAG/
├── main.py                 # FastAPI entry
├── templates/index.html    # Primary UI
├── src/
│   ├── api/routes.py       # Upload, chat, clear
│   ├── config.py           # Settings
│   ├── ollama/client.py    # Ollama HTTP client
│   └── rag/                # Retrieval pipeline (preserved)
├── vector_db/              # Chroma persistence (keep)
├── tests/
├── archive/                # Parked Streamlit + research agent
├── deploy/                 # Optional Helm/K8s/Terraform/CI
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## Attribution

G-RAG is a personalized refactor of a prior open local RAG research-assistant codebase.
Third-party libraries retain their own licenses. Optional deployment assets live under `deploy/`.
