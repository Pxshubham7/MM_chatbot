import base64
import logging
from typing import Dict, List, Optional
import httpx
from backend.app.config import get_settings
from backend.app.providers.base import (
    LLMProvider,
    EmbeddingProvider,
    VisionAnalyzer,
    AudioTranscriber,
    TTSProvider,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class OpenAILLMProvider(LLMProvider):
    """OpenAI-compatible LLM dialogue provider using HTTP client."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.openai_api_key
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.model = model or settings.openai_llm_model

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        context: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not configured in .env")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": 0.6,
            "max_tokens": 800
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                return data["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.error(f"OpenAI LLM request failed: {exc}")
            raise


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible text embedding provider."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.openai_api_key
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.model = model or settings.openai_embedding_model

    @property
    def dimension(self) -> int:
        return 1536

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not configured in .env")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": texts
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(f"{self.base_url}/embeddings", headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                return [item["embedding"] for item in data["data"]]
        except Exception as exc:
            logger.error(f"OpenAI Embedding request failed: {exc}")
            raise


class OpenAIVisionAnalyzer(VisionAnalyzer):
    """OpenAI-compatible multimodal vision provider."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.openai_api_key
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.model = model or "gpt-4o-mini"

    def describe_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        user_prompt: Optional[str] = None
    ) -> str:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not configured in .env")

        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        prompt = (
            user_prompt or
            "Please describe the content and emotional atmosphere of this image objectively and supportively."
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                return data["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.error(f"OpenAI Vision request failed: {exc}")
            raise


class OpenAIAudioTranscriber(AudioTranscriber):
    """OpenAI Whisper speech-to-text audio transcriber."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or settings.openai_api_key
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")

    def transcribe_audio(self, audio_bytes: bytes, mime_type: str) -> str:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not configured in .env")

        headers = {"Authorization": f"Bearer {self.api_key}"}
        ext = "wav" if "wav" in mime_type else "mp3"
        files = {
            "file": (f"audio.{ext}", audio_bytes, mime_type),
            "model": (None, "whisper-1")
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                res = client.post(f"{self.base_url}/audio/transcriptions", headers=headers, files=files)
                res.raise_for_status()
                return res.json()["text"]
        except Exception as exc:
            logger.error(f"OpenAI Whisper request failed: {exc}")
            raise


class OpenAITTSProvider(TTSProvider):
    """OpenAI text-to-speech audio provider."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or settings.openai_api_key
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")

    def synthesize_speech(self, text: str) -> bytes:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not configured in .env")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "tts-1",
            "voice": "alloy",
            "input": text[:1000]  # Safe length limit
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(f"{self.base_url}/audio/speech", headers=headers, json=payload)
                res.raise_for_status()
                return res.content
        except Exception as exc:
            logger.error(f"OpenAI TTS request failed: {exc}")
            raise
