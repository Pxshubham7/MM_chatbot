import math
import struct
import wave
import io
import re
from typing import Dict, List, Optional
from backend.app.providers.base import (
    LLMProvider,
    EmbeddingProvider,
    VisionAnalyzer,
    AudioTranscriber,
    TTSProvider,
)


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic offline embedding generator.
    Creates 64-dimensional unit-length vectors based on term frequencies and semantic hashes.
    Guarantees meaningful cosine similarity scores for relevant search terms without calling external APIs.
    """
    def __init__(self, dimension: int = 64):
        self._dimension = dimension
        # Semantic keyword anchor clusters to give semantic relevance in mock mode
        self.keywords = [
            "breath", "inhale", "exhale", "lungs", "box", "4-7-8", "air", "pace",
            "ground", "sensory", "5-4-3-2-1", "sight", "touch", "hear", "smell", "taste",
            "sleep", "rest", "bed", "insomnia", "night", "routine", "pillow", "dark",
            "anxiety", "panic", "overwhelm", "stress", "calm", "relax", "tension", "muscle",
            "emotion", "feeling", "distress", "cope", "support", "listen", "heart", "peace",
            "journal", "write", "mind", "focus", "present", "reset", "release", "warmth"
        ]

    @property
    def dimension(self) -> int:
        return self._dimension

    def _text_to_vector(self, text: str) -> List[float]:
        text_lower = text.lower()
        vec = [0.0] * self._dimension

        # 1. Base keyword projection
        for i, kw in enumerate(self.keywords):
            if kw in text_lower:
                vec[i % self._dimension] += 2.5

        # 2. Character n-gram hashing for vocabulary generalization
        tokens = re.findall(r"\w+", text_lower)
        for token in tokens:
            h = hash(token)
            idx = abs(h) % self._dimension
            sign = 1.0 if (h // self._dimension) % 2 == 0 else -1.0
            vec[idx] += sign * (1.0 / (1.0 + math.log(len(token) + 1)))

        # 3. L2 Normalize vector
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return [round(v / norm, 6) for v in vec]

    def embed_text(self, text: str) -> List[float]:
        return self._text_to_vector(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._text_to_vector(t) for t in texts]


class MockLLMProvider(LLMProvider):
    """
    Offline mock LLM that strictly observes non-clinical boundaries,
    acknowledges retrieved context, and references sources.
    """
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        context: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        latest_message = messages[-1]["content"] if messages else ""
        
        # Check if context was retrieved
        if context and context.strip():
            response = (
                f"I hear how challenging this moment is, and I want to support you gently. "
                f"Based on our curated wellness resources, here is a practical exercise that might help right now:\n\n"
                f"{context.strip()}\n\n"
                f"Take this at your own pace without any pressure. "
                f"Would you like to try one of these steps together or simply share more about how you are feeling?"
            )
            return response
        else:
            response = (
                f"Thank you for sharing that with me. I want you to know that your feelings are valid, "
                f"and I'm here to listen without judgment.\n\n"
                f"*Note: Our curated knowledge base does not currently contain specific documentation "
                f"directly matching this topic.* However, taking a slow, deep breath and giving yourself "
                f"a moment of space can always be a helpful starting point. "
                f"How has your body been feeling while carrying this?"
            )
            return response


class MockVisionAnalyzer(VisionAnalyzer):
    """Offline vision analyzer returning structured non-clinical visual context."""
    def describe_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        user_prompt: Optional[str] = None
    ) -> str:
        size_kb = len(image_bytes) / 1024
        return (
            f"[Visual Analysis ({mime_type}, {size_kb:.1f} KB): "
            f"The uploaded image was processed. It appears to depict an everyday scene or object. "
            f"No distressing, clinical, or acute hazard elements were detected.]"
        )


class MockAudioTranscriber(AudioTranscriber):
    """Offline audio transcriber returning mock transcription."""
    def transcribe_audio(self, audio_bytes: bytes, mime_type: str) -> str:
        size_kb = len(audio_bytes) / 1024
        return f"I've been feeling overwhelmed lately and could really use a moment to calm down. (Audio {size_kb:.1f}KB)"


class MockTTSProvider(TTSProvider):
    """
    Generates a valid, minimal PCM WAV audio file (sine wave tone/silence)
    so that client audio players receive real, playable audio bytes offline.
    """
    def synthesize_speech(self, text: str) -> bytes:
        sample_rate = 16000
        duration_seconds = 1.0  # 1 second gentle tone/silence indicator
        num_samples = int(sample_rate * duration_seconds)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)

            # Generate a very soft 220Hz (A3) gentle tone fading to silence
            frames = bytearray()
            for i in range(num_samples):
                t = float(i) / sample_rate
                decay = math.exp(-3.0 * t)
                sample_val = int(1500.0 * math.sin(2.0 * math.pi * 220.0 * t) * decay)
                frames.extend(struct.pack("<h", sample_val))

            wav_file.writeframes(frames)

        return buffer.getvalue()
