import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import (
    FastAPI,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.app.config import get_settings
from backend.app.db.session import get_db, init_db, get_db_status
from backend.app.guardrails.prompts import SYSTEM_PROMPT, format_rag_context
from backend.app.guardrails.safety import SafetyGuardrail
from backend.app.providers.factory import ProviderFactory
from backend.app.rag.retriever import RAGRetriever
from backend.app.rag.store import VectorStoreAdapter
from backend.app.schemas import (
    ChatResponse,
    HealthResponse,
    IngestResponse,
)
from backend.app.utils.media import (
    validate_and_read_image,
    validate_and_read_audio,
)

# Configure privacy-safe logging (no request bodies or secrets logged)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("mm_chatbot.api")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown."""
    logger.info("Initializing RAG database schemas...")
    init_db()
    logger.info(f"API backend started in '{settings.app_env}' environment (provider_mode: {settings.provider_mode}).")
    yield
    logger.info("API backend shutting down.")


app = FastAPI(
    title="Multimodal Emotional Support Chatbot API",
    description="Privacy-conscious, non-clinical emotional support chatbot with RAG and multimodal capabilities.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class PrivacyAuditMiddleware:
    """
    Pure ASGI privacy-safe audit logging middleware.
    Avoids BaseHTTPMiddleware deadlock issues in ASGI test runners.
    Logs HTTP method, path, response status, and duration without logging payloads.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        start_time = time.time()
        status_code = [200]

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code[0] = message.get("status", 200)
            await send(message)

        await self.app(scope, receive, send_wrapper)
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            f"{scope['method']} {scope['path']} -> Status: {status_code[0]} ({duration_ms:.1f}ms)"
        )


app.add_middleware(PrivacyAuditMiddleware)


@app.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    """Health check endpoint confirming API status, DB connectivity, and indexed documents count."""
    status_info = get_db_status()
    store = VectorStoreAdapter(db)
    doc_count = store.count_documents() if status_info["connected"] else 0

    return HealthResponse(
        status="healthy" if status_info["connected"] else "degraded",
        database_type=status_info.get("type", "unknown"),
        database_connected=status_info.get("connected", False),
        provider_mode=settings.provider_mode,
        documents_indexed=doc_count
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(
    message: Optional[str] = Form(None),
    conversation_history_json: Optional[str] = Form(None),
    generate_speech: bool = Form(False),
    image_file: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """
    Multimodal chat endpoint accepting text, image upload, and audio upload.
    Orchestrates transcription, vision description, emotional support guardrails,
    RAG retrieval over MySQL, LLM synthesis, and optional speech generation.
    """
    # Parse generate_speech boolean flag safely (handles bool, str, Form)
    is_tts = False
    if generate_speech is True or (isinstance(generate_speech, str) and generate_speech.lower() in ("true", "1", "yes")):
        is_tts = True

    # 1. Media Validation & Processing
    image_bytes, image_mime = validate_and_read_image(image_file) if image_file else (None, None)
    audio_bytes, audio_mime = validate_and_read_audio(audio_file) if audio_file else (None, None)

    # 2. Instantiate Providers via Factory
    llm = ProviderFactory.get_llm_provider()
    embedding_provider = ProviderFactory.get_embedding_provider()
    vision = ProviderFactory.get_vision_analyzer()
    stt = ProviderFactory.get_audio_transcriber()
    tts = ProviderFactory.get_tts_provider()

    transcribed_text: Optional[str] = None
    image_description: Optional[str] = None

    # Handle Audio Input (Speech-to-Text)
    if audio_bytes:
        transcribed_text = stt.transcribe_audio(audio_bytes, audio_mime)

    # Handle Image Input (Vision Context)
    if image_bytes:
        image_description = vision.describe_image(image_bytes, image_mime, user_prompt=message)

    # Assemble complete user prompt
    text_content = (message or "").strip()
    if transcribed_text:
        text_content = f"{text_content}\n[Spoken Audio Transcription]: {transcribed_text}".strip()
    if image_description:
        text_content = f"{text_content}\n{image_description}".strip()

    if not text_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No input provided. Please provide text, an image, or an audio recording."
        )

    # 3. Guardrail & Safety Check (Crisis Triage Interceptor)
    if settings.enable_crisis_guardrail:
        is_crisis, crisis_reply = SafetyGuardrail.check_crisis(text_content)
        if is_crisis and crisis_reply:
            return ChatResponse(
                answer=crisis_reply,
                citations=[],
                has_rag_context=False,
                is_crisis_response=True,
                transcribed_text=transcribed_text,
                image_description=image_description
            )

    # Sanitize user input against prompt injection
    safe_user_prompt = SafetyGuardrail.sanitize_user_input(text_content)

    # 4. RAG Retrieval over Curated Knowledge Base
    retriever = RAGRetriever(db=db, embedding_provider=embedding_provider)
    search_query = message if message and message.strip() else (transcribed_text or text_content)
    matches, citations = retriever.retrieve(search_query)

    has_rag_context = len(matches) > 0
    rag_context_str = format_rag_context(matches) if has_rag_context else ""

    # 5. Format Ephemeral Dialogue History
    formatted_messages = []
    if conversation_history_json and isinstance(conversation_history_json, str):
        try:
            history = json.loads(conversation_history_json)
            for turn in history[-6:]:  # Keep active session window manageable
                if isinstance(turn, dict) and "role" in turn and "content" in turn:
                    formatted_messages.append({
                        "role": turn["role"],
                        "content": SafetyGuardrail.sanitize_user_input(turn["content"])
                    })
        except Exception as exc:
            logger.warning(f"Failed to parse conversation_history_json: {exc}")

    # Add current user query to messages
    formatted_messages.append({"role": "user", "content": safe_user_prompt})

    # 6. LLM Response Generation
    raw_answer = llm.generate_response(
        messages=formatted_messages,
        context=rag_context_str,
        system_prompt=SYSTEM_PROMPT
    )

    # Post-process response to ensure no clinical claims or diagnoses
    final_answer = SafetyGuardrail.audit_model_output(raw_answer)

    # 7. Optional Spoken Audio Synthesis
    audio_base64_str: Optional[str] = None
    if is_tts:
        try:
            speech_bytes = tts.synthesize_speech(final_answer)
            audio_base64_str = base64.b64encode(speech_bytes).decode("utf-8")
        except Exception as exc:
            logger.error(f"Text-to-speech synthesis failed: {exc}")

    # 8. Return response (Temporary media buffers are discarded by garbage collector)
    return ChatResponse(
        answer=final_answer,
        citations=citations,
        has_rag_context=has_rag_context,
        is_crisis_response=False,
        audio_base64=audio_base64_str,
        audio_format="audio/wav",
        transcribed_text=transcribed_text,
        image_description=image_description
    )


@app.post("/api/ingest", response_model=IngestResponse)
def trigger_ingest(docs_dir: str = "data/sample_docs", db: Session = Depends(get_db)):
    """Triggers ingestion of curated local documents."""
    from backend.scripts.ingest import ingest_directory
    try:
        ingest_directory(docs_dir)
        store = VectorStoreAdapter(db)
        return IngestResponse(
            documents_ingested=store.count_documents(),
            chunks_ingested=store.count_chunks(),
            message="Curated documents ingested successfully."
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
