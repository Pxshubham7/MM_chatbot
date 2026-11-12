import pytest
from fastapi import HTTPException
from backend.app.db.session import get_db
from backend.app.main import health_check, chat_endpoint
from backend.app.utils.media import validate_and_read_image, validate_and_read_audio


@pytest.fixture
def db_session():
    gen = get_db()
    db = next(gen)
    yield db
    gen.close()


def test_health_endpoint(db_session):
    res = health_check(db=db_session)
    assert res.status in ["healthy", "degraded"]
    assert res.database_connected is True
    assert res.documents_indexed >= 3
    assert res.provider_mode == "mock"


def test_chat_endpoint_normal_query(db_session):
    res = chat_endpoint(
        message="Can you explain how to practice 5-4-3-2-1 sensory grounding?",
        generate_speech=False,
        image_file=None,
        audio_file=None,
        db=db_session
    )
    assert res.is_crisis_response is False
    assert res.has_rag_context is True
    assert len(res.citations) > 0
    assert any("grounding" in cit.category.lower() for cit in res.citations)
    assert "curated wellness resources" in res.answer


def test_chat_endpoint_crisis_interception(db_session):
    res = chat_endpoint(
        message="I don't want to live anymore, I want to kill myself",
        generate_speech=False,
        image_file=None,
        audio_file=None,
        db=db_session
    )
    assert res.is_crisis_response is True
    assert "local emergency services" in res.answer
    assert len(res.citations) == 0


def test_chat_endpoint_with_tts(db_session):
    res = chat_endpoint(
        message="Hello, take a slow deep breath.",
        generate_speech=True,
        image_file=None,
        audio_file=None,
        db=db_session
    )
    assert res.audio_base64 is not None
    assert len(res.audio_base64) > 100
    assert res.audio_format == "audio/wav"


def test_chat_endpoint_media_validation():
    class DummyFile:
        def __init__(self, filename, content_type, content=b"test"):
            self.filename = filename
            self.content_type = content_type
            import io
            self.file = io.BytesIO(content)

    # Unsupported format
    bad_file = DummyFile("bad.exe", "application/x-msdownload")
    with pytest.raises(HTTPException) as exc:
        validate_and_read_image(bad_file)
    assert exc.value.status_code == 400

    # Oversized image
    huge_file = DummyFile("large.png", "image/png", b"x" * (11 * 1024 * 1024))
    with pytest.raises(HTTPException) as exc:
        validate_and_read_image(huge_file)
    assert exc.value.status_code == 413
