import logging
from backend.app.config import get_settings
from backend.app.providers.base import (
    LLMProvider,
    EmbeddingProvider,
    VisionAnalyzer,
    AudioTranscriber,
    TTSProvider,
)
from backend.app.providers.mock import (
    MockLLMProvider,
    MockEmbeddingProvider,
    MockVisionAnalyzer,
    MockAudioTranscriber,
    MockTTSProvider,
)
from backend.app.providers.openai_provider import (
    OpenAILLMProvider,
    OpenAIEmbeddingProvider,
    OpenAIVisionAnalyzer,
    OpenAIAudioTranscriber,
    OpenAITTSProvider,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class ProviderFactory:
    """
    Factory for instantiating concrete multimodal providers based on configuration.
    Keeps all models, embeddings, vision, transcription, and speech synthesis
    behind replaceable interfaces.
    """

    @staticmethod
    def get_llm_provider() -> LLMProvider:
        mode = settings.provider_mode.lower()
        if mode == "openai" and settings.openai_api_key:
            return OpenAILLMProvider()
        elif mode == "openai" and not settings.openai_api_key:
            logger.warning("OpenAI requested but OPENAI_API_KEY is not set. Falling back to MockLLMProvider.")
        return MockLLMProvider()

    @staticmethod
    def get_embedding_provider() -> EmbeddingProvider:
        mode = settings.provider_mode.lower()
        if mode == "openai" and settings.openai_api_key:
            return OpenAIEmbeddingProvider()
        elif mode == "openai" and not settings.openai_api_key:
            logger.warning("OpenAI requested but OPENAI_API_KEY is not set. Falling back to MockEmbeddingProvider.")
        return MockEmbeddingProvider()

    @staticmethod
    def get_vision_analyzer() -> VisionAnalyzer:
        mode = settings.provider_mode.lower()
        if mode == "openai" and settings.openai_api_key:
            return OpenAIVisionAnalyzer()
        return MockVisionAnalyzer()

    @staticmethod
    def get_audio_transcriber() -> AudioTranscriber:
        mode = settings.provider_mode.lower()
        if mode == "openai" and settings.openai_api_key:
            return OpenAIAudioTranscriber()
        return MockAudioTranscriber()

    @staticmethod
    def get_tts_provider() -> TTSProvider:
        mode = settings.provider_mode.lower()
        if mode == "openai" and settings.openai_api_key:
            return OpenAITTSProvider()
        return MockTTSProvider()
