# Multimodal Emotional-Support Chatbot (RAG & Guardrails)

A secure, privacy-conscious chatbot featuring multimodal input and output, retrieval-augmented generation (RAG) over MySQL 8.0, and a non-clinical emotional-support experience.

---

## Architecture Overview

```
[ Trusted Local User ]
         │
         ▼
 ┌──────────────────────┐
 │ Streamlit UI (8501)  │  (Zero DB credentials, zero provider secrets;
 │ frontend/app.py      │   ephemeral chat history in st.session_state only)
 └──────────┬───────────┘
            │ HTTP Multipart Form
            ▼
 ┌──────────────────────┐
 │ FastAPI API (8000)   │
 │ backend/app/main.py  │
 │  ├── Media Validator │  (MIME type & size checks, streaming cleanup)
 │  ├── Safety Engine   │  (Deterministic crisis triage, prompt injection defense)
 │  ├── RAG Retriever   │  (Cosine similarity calculation application-side)
 │  └── Provider Layer  │  (Replaceable ABC adapters: LLM, STT, Vision, TTS, Embeddings)
 └──────────┬───────────┘
            │ SQLAlchemy (PyMySQL)
            ▼
 ┌──────────────────────┐
 │ MySQL 8.0 Database   │
 │  ├── documents       │  (Curated coping guides metadata)
 │  └── document_chunks │  (Curated chunks + serialized vector embeddings in JSON)
 └──────────────────────┘
            ▲
            │ Ingestion CLI
 ┌──────────┴───────────┐
 │ backend/scripts/     │
 │ ingest.py            │  (Admin-free local ingestion tool)
 └──────────────────────┘
```

### Key Architectural Boundaries
- **Streamlit communicates strictly with FastAPI**: The browser-facing frontend never opens a direct database connection and never holds database credentials or cloud API keys.
- **Provider Abstraction**: All LLMs, embeddings, image understanding, speech-to-text, and text-to-speech run behind abstract base classes (`LLMProvider`, `EmbeddingProvider`, `VisionAnalyzer`, `AudioTranscriber`, `TTSProvider`).
- **MySQL 8.0 Vector Storage Adapter**: Because MySQL 8.0 does not include the native `VECTOR` data type (introduced in MySQL 8.4+), embeddings are stored in `document_chunks.embedding` as JSON float arrays. The `VectorStoreAdapter` executes fast application-side cosine similarity using NumPy.
- **Scaling Limits**: In-memory similarity calculation performs under 15ms for up to 5,000 chunks. Beyond 50,000 chunks, migrating to MySQL 8.4+ native `VECTOR` indexes or an external vector database is recommended.

---

## Security & Privacy Model

- **Single Trusted User**: Designed for one trusted user operating locally (`127.0.0.1`) or inside an isolated private network. It does not provide user-level tenancy separation or authentication.
- **No End-to-End Encryption**: Responses require server-side context processing. All media and chat turns are processed in temporary server memory.
- **Zero Persistent Conversation Logs**: No raw chat messages, user prompts, uploaded images, or audio recordings are saved to MySQL or backend disk. Data exists ephemerally only in the user's active browser tab (`st.session_state`).
- **Privacy-Safe Audit Logging**: The backend logging middleware explicitly suppresses request payloads, image/audio bytes, and headers, recording only HTTP method, endpoint path, status code, and duration.
- **Prompt Injection Defense**: Retrieved RAG documents and user inputs are strictly encapsulated within untrusted XML boundaries (`<retrieved_context>` / `<user_query>`), preventing untrusted data from overriding safety rules.
- **Private Network Notice**: If this service is accessed over a local network, it must be protected behind a reverse proxy (e.g. Nginx or Caddy) with TLS/HTTPS enabled and firewall restrictions limiting access to trusted IP addresses.

---

## Emotional-Support Guardrails

- **Non-Clinical Persona**: The chatbot is an empathetic conversational guide designed for everyday stress, reflection, and mindful coping. It **never** diagnoses psychiatric conditions, prescribes medications, claims to be a therapist, or replaces licensed medical care.
- **Deterministic Crisis Intervention**: If acute self-harm, suicidal ideation, or immediate danger is detected, the safety engine intercepts the query before calling the LLM or RAG pipeline. It immediately returns fixed, calm, non-judgmental guidance directing the user to local emergency services without inventing unverified local hotline numbers.
- **Grounding & Citations**: When suggestions are grounded in curated documents, source citations are displayed. If no relevant documents match, the assistant transparently clarifies that the knowledge base does not cover the specific topic while offering general reflective support.

---

## Supported Multimodal Inputs & Limits

| Media Type | Allowed MIME Types | Maximum File Size | Processing Method |
| :--- | :--- | :--- | :--- |
| **Text** | Plain text | 4,000 characters | Direct LLM prompt assembly |
| **Images** | `image/jpeg`, `image/png`, `image/webp` | 10 MB | Multimodal vision adapter |
| **Audio** | `audio/wav`, `audio/mpeg`, `audio/mp3`, `audio/ogg` | 25 MB | Speech-to-text transcriber |
| **Output Audio** | `audio/wav` | N/A | TTS provider (optional toggle) |

---

## Installation & Setup

### 1. Prerequisites
- Python 3.11+ (Python 3.13 verified)
- MySQL 8.0 Server (or SQLite fallback)

### 2. Environment Configuration
Copy the environment template and configure database credentials:
```powershell
cp .env.example .env
```
Key settings in `.env`:
```ini
APP_ENV=development
API_HOST=127.0.0.1
API_PORT=8000
DATABASE_URL=mysql+pymysql://root:password@127.0.0.1:3306/mm_chatbot_rag
ALLOW_SQLITE_FALLBACK=true
PROVIDER_MODE=mock   # Change to "openai" to use OpenAI API with OPENAI_API_KEY
RAG_SIMILARITY_THRESHOLD=0.25
MAX_IMAGE_SIZE_MB=10
MAX_AUDIO_SIZE_MB=25
```

### 3. Initialize Database & Tables
Run the database schema initialization CLI:
```powershell
.\.venv\Scripts\python.exe -m backend.scripts.init_db
```

### 4. Ingest Curated Knowledge Documents
Ingest the curated non-clinical coping guides:
```powershell
.\.venv\Scripts\python.exe -m backend.scripts.ingest --dir data/sample_docs
```

---

## Running the Application

### Start the FastAPI Backend
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation (Swagger UI): `http://127.0.0.1:8000/docs`

### Start the Streamlit Frontend
In a separate terminal:
```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8501
```
Open your browser at `http://localhost:8501`.

---

## Running Automated Tests

Run the full pytest test suite (16 tests covering guardrails, vector store, API endpoints, and mock providers):
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```
