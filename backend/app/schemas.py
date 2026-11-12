from typing import List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Ephemeral chat message item passed from client session state."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class CitationItem(BaseModel):
    """Source citation from curated document collection."""
    document_title: str = Field(..., description="Document title")
    source_uri: str = Field(..., description="Local source document URI/path")
    category: str = Field(..., description="Document category or theme")
    chunk_index: int = Field(..., description="Zero-indexed chunk position in document")
    content_excerpt: str = Field(..., description="Relevant text excerpt supporting the response")
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 - 1.0)")


class ChatResponse(BaseModel):
    """Standard response from the chatbot endpoint."""
    answer: str = Field(..., description="Grounded non-clinical response text")
    citations: List[CitationItem] = Field(default_factory=list, description="Grounding source citations")
    has_rag_context: bool = Field(default=False, description="True if knowledge base context supported the answer")
    is_crisis_response: bool = Field(default=False, description="True if crisis intervention guardrail was triggered")
    audio_base64: Optional[str] = Field(default=None, description="Optional base64-encoded spoken response audio")
    audio_format: Optional[str] = Field(default="audio/wav", description="Audio MIME format, e.g. audio/wav")
    transcribed_text: Optional[str] = Field(default=None, description="Transcribed text if audio was uploaded")
    image_description: Optional[str] = Field(default=None, description="Extracted visual context if image was uploaded")


class HealthResponse(BaseModel):
    """Application and database health check status."""
    status: str
    database_type: str
    database_connected: bool
    provider_mode: str
    documents_indexed: int


class IngestResponse(BaseModel):
    """Result of document ingestion operation."""
    documents_ingested: int
    chunks_ingested: int
    message: str
