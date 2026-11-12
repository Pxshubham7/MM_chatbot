import io
from typing import Optional, Tuple
from fastapi import HTTPException, UploadFile, status
from backend.app.config import get_settings

settings = get_settings()

ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

ALLOWED_AUDIO_TYPES = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
}


def validate_and_read_image(file: Optional[UploadFile]) -> Optional[Tuple[bytes, str]]:
    """
    Validates uploaded image file type and size.
    Returns (image_bytes, mime_type) or None if no file was uploaded.
    Raises HTTPException on validation failure.
    """
    if file is None or not hasattr(file, "filename") or not file.filename:
        return None

    content_type = (getattr(file, "content_type", "") or "").lower()
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image format '{content_type}'. Allowed: {list(ALLOWED_IMAGE_TYPES.keys())}"
        )

    if hasattr(file, "file") and hasattr(file.file, "read"):
        content = file.file.read()
    else:
        content = b""

    max_bytes = settings.max_image_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image file exceeds maximum allowed size of {settings.max_image_size_mb} MB"
        )

    return content, content_type


def validate_and_read_audio(file: Optional[UploadFile]) -> Optional[Tuple[bytes, str]]:
    """
    Validates uploaded audio file type and size.
    Returns (audio_bytes, mime_type) or None if no file was uploaded.
    Raises HTTPException on validation failure.
    """
    if file is None or not hasattr(file, "filename") or not file.filename:
        return None

    content_type = (getattr(file, "content_type", "") or "").lower()
    if content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format '{content_type}'. Allowed: {list(ALLOWED_AUDIO_TYPES.keys())}"
        )

    if hasattr(file, "file") and hasattr(file.file, "read"):
        content = file.file.read()
    else:
        content = b""

    max_bytes = settings.max_audio_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file exceeds maximum allowed size of {settings.max_audio_size_mb} MB"
        )

    return content, content_type
