from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables and .env file.
    All secrets are kept backend-side and never exposed to the client UI.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Server Settings
    app_env: str = Field(default="development", description="Environment: development, test, production")
    api_host: str = Field(default="127.0.0.1", description="FastAPI host binding")
    api_port: int = Field(default=8000, description="FastAPI port binding")
    cors_origins: str = Field(
        default="http://localhost:8501,http://127.0.0.1:8501",
        description="Comma-separated allowed origins for Streamlit UI"
    )

    # Database Configuration (MySQL 8.0 with SQLite fallback for offline development/tests)
    database_url: str = Field(
        default="mysql+pymysql://root:password@127.0.0.1:3306/mm_chatbot_rag",
        description="MySQL 8.0 connection URL"
    )
    allow_sqlite_fallback: bool = Field(
        default=True,
        description="Allow automatic fallback to local SQLite if MySQL is unreachable during tests or offline local run"
    )
    sqlite_db_path: str = Field(
        default="sqlite:///./mm_chatbot_rag.db",
        description="Fallback SQLite database path"
    )

    # Provider Configuration
    # Options: "mock" (offline testing with zero external API calls/costs), "openai", "gemini"
    provider_mode: str = Field(
        default="mock",
        description="Active provider backend for LLM, vision, STT, and embeddings"
    )

    # Provider Credentials & Model Identifiers
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API Key")
    openai_base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI API Base URL")
    openai_llm_model: str = Field(default="gpt-4o-mini", description="OpenAI LLM Model Name")
    openai_embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI Embedding Model")

    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API Key")
    gemini_llm_model: str = Field(default="gemini-1.5-flash", description="Gemini LLM Model")

    # Media Limits
    max_image_size_mb: int = Field(default=10, description="Maximum allowed image size in MB")
    max_audio_size_mb: int = Field(default=25, description="Maximum allowed audio size in MB")

    # RAG Settings
    rag_similarity_threshold: float = Field(default=0.25, description="Minimum cosine similarity for document retrieval")
    rag_top_k: int = Field(default=3, description="Maximum number of context chunks to retrieve")

    # Guardrails
    enable_crisis_guardrail: bool = Field(default=True, description="Enable deterministic crisis triage interceptor")

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache()
def get_settings() -> Settings:
    """Returns cached application settings instance."""
    return Settings()
