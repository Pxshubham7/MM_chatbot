from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class LLMProvider(ABC):
    """Abstract interface for large language model dialogue synthesis."""

    @abstractmethod
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        context: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generates a non-clinical, empathetic response given message history and retrieved context.
        """
        pass


class EmbeddingProvider(ABC):
    """Abstract interface for generating vector embeddings."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generates embedding vector for a single text chunk or query."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a batch of text chunks."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the dimensionality of the generated vectors."""
        pass


class VisionAnalyzer(ABC):
    """Abstract interface for image understanding and visual context extraction."""

    @abstractmethod
    def describe_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        user_prompt: Optional[str] = None
    ) -> str:
        """
        Analyzes an uploaded image and returns an objective, supportive visual description.
        """
        pass


class AudioTranscriber(ABC):
    """Abstract interface for speech-to-text audio transcription."""

    @abstractmethod
    def transcribe_audio(self, audio_bytes: bytes, mime_type: str) -> str:
        """
        Transcribes uploaded audio bytes to text.
        """
        pass


class TTSProvider(ABC):
    """Abstract interface for text-to-speech audio synthesis."""

    @abstractmethod
    def synthesize_speech(self, text: str) -> bytes:
        """
        Converts response text to spoken audio bytes (WAV or MP3).
        """
        pass
