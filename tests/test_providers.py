import pytest
from backend.app.providers.mock import (
    MockLLMProvider,
    MockEmbeddingProvider,
    MockVisionAnalyzer,
    MockAudioTranscriber,
    MockTTSProvider,
)


def test_mock_embedding_properties():
    provider = MockEmbeddingProvider(dimension=64)
    assert provider.dimension == 64
    vec = provider.embed_text("Grounding exercise for stress and anxiety")
    assert len(vec) == 64
    # Check that vector is normalized to approx unit length
    magnitude = sum(x * x for x in vec) ** 0.5
    assert pytest.approx(magnitude, 0.01) == 1.0


def test_mock_llm_with_context():
    provider = MockLLMProvider()
    messages = [{"role": "user", "content": "How do I do box breathing?"}]
    context = "Box breathing: Inhale 4s, hold 4s, exhale 4s."
    res = provider.generate_response(messages, context=context)
    assert "Box breathing" in res
    assert "curated wellness resources" in res


def test_mock_llm_without_context():
    provider = MockLLMProvider()
    messages = [{"role": "user", "content": "What is the capital of Mars?"}]
    res = provider.generate_response(messages, context="")
    assert "knowledge base does not currently contain specific documentation" in res


def test_mock_tts_valid_wav():
    provider = MockTTSProvider()
    audio_bytes = provider.synthesize_speech("Hello, take a slow deep breath.")
    assert len(audio_bytes) > 100
    # Check WAV header signature ('RIFF' and 'WAVE')
    assert audio_bytes[:4] == b"RIFF"
    assert audio_bytes[8:12] == b"WAVE"


def test_mock_vision_and_audio():
    vision = MockVisionAnalyzer()
    desc = vision.describe_image(b"fake_image_bytes", "image/png")
    assert "Visual Analysis" in desc

    stt = MockAudioTranscriber()
    transcript = stt.transcribe_audio(b"fake_audio_bytes", "audio/wav")
    assert "overwhelmed" in transcript
