"""
Audio Pipeline — OpenAI-совместимый эндпоинт для транскрипции аудио.

Маршруты:
  POST /v1/audio/transcriptions  — транскрипция аудиофайла (ASR)
  POST /v1/audio/chat            — загрузка аудио + авто-ответ ассистента

OpenWebUI интеграция:
  Admin Panel → Audio → STT Engine: OpenAI
  STT Base URL: http://backend:8000/v1
  STT API Key: <MWS_API_KEY>
  STT Model: whisper-turbo-local

Поддерживаемые форматы: webm, mp3, wav, ogg, m4a, flac, mp4
Лимит файла: 25 MB
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.services.mws_client import MWSAPIError, transcribe_audio

logger = logging.getLogger("mts.audio")

router = APIRouter()

# Разрешённые MIME-типы
ALLOWED_CONTENT_TYPES = {
    "audio/webm",
    "audio/mpeg",   # mp3
    "audio/mp3",
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/ogg",
    "audio/m4a",
    "audio/mp4",
    "audio/flac",
    "video/webm",   # Браузеры часто шлют webm как video/
    "video/mp4",
    "application/octet-stream",  # Fallback
}

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB


# POST /v1/audio/transcriptions — OpenAI-compatible ASR endpoint

@router.post("/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(..., description="Аудиофайл для транскрипции"),
    model: Optional[str] = Form(default=None, description="Модель ASR (игнорируется, используется whisper)"),
    language: Optional[str] = Form(default=None, description="Код языка, например 'ru'"),
    response_format: Optional[str] = Form(default="json"),
    timestamp_granularities: Optional[str] = Form(default=None),
):
    """
    OpenAI-совместимый эндпоинт для транскрипции аудио.

    Принимает аудиофайл и возвращает распознанный текст.
    Используется OpenWebUI для голосового ввода (микрофон).

    Returns:
        {"text": "...", "language": "ru", "duration": null}
    """
    logger.info(
        "audio/transcriptions: filename=%r content_type=%r",
        file.filename,
        file.content_type,
    )

    # Валидация Content-Type
    content_type = (file.content_type or "application/octet-stream").split(";")[0].strip()
    if content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning("Unsupported audio content_type: %s", content_type)
        # Не блокируем — MWS сам разберётся с форматом

    # Читаем файл с проверкой размера
    try:
        audio_bytes = await file.read()
    except Exception as e:
        logger.error("Failed to read uploaded file: %s", e)
        raise HTTPException(status_code=400, detail=f"Ошибка чтения файла: {e}")

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Файл пустой.")

    if len(audio_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Файл слишком большой ({len(audio_bytes) // 1024 // 1024}MB). Максимум: 25MB.",
        )

    logger.info("Transcribing audio: %d bytes", len(audio_bytes))

    # Вызов MWS Whisper API
    try:
        text = await transcribe_audio(audio_bytes)
    except MWSAPIError as e:
        logger.error("ASR MWSAPIError: %s", e)
        raise HTTPException(status_code=502, detail=f"Ошибка ASR API: {e}")
    except Exception:
        logger.exception("ASR unexpected error")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка транскрипции.")

    logger.info("Transcription result: %r", text[:100] if text else "(empty)")

    # OpenAI-совместимый ответ
    return JSONResponse(
        content={
            "text": text,
            "language": language or "ru",
            "duration": None,
            "task": "transcribe",
        }
    )


# POST /v1/audio/chat — аудио → ASR → ответ ассистента

@router.post("/audio/chat")
async def audio_chat(
    file: UploadFile = File(..., description="Аудиофайл с вопросом"),
    user_id: Optional[str] = Form(default=None),
    model: Optional[str] = Form(default=None),
):
    """
    Голосовой чат-эндпоинт: загружает аудио, транскрибирует, получает ответ.

    Возвращает:
        {"transcript": "...", "reply": "...", "model": "..."}

    Примечание: OpenWebUI не использует этот эндпоинт напрямую.
    Он полезен для прямой интеграции кастомных клиентов.
    """
    uid = user_id or "anonymous"

    # Читаем файл
    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения файла: {e}")

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Файл пустой.")

    if len(audio_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Файл слишком большой. Максимум: 25MB.")

    # Транскрипция
    try:
        transcript = await transcribe_audio(audio_bytes)
    except MWSAPIError as e:
        raise HTTPException(status_code=502, detail=f"Ошибка ASR: {e}")
    except Exception:
        logger.exception("audio_chat: transcription failed")
        raise HTTPException(status_code=500, detail="Ошибка транскрипции.")

    if not transcript or not transcript.strip():
        return JSONResponse(
            content={
                "transcript": "",
                "reply": "Не удалось распознать речь. Пожалуйста, говорите чётче или используйте другой формат аудио.",
                "model": "whisper",
            }
        )

    logger.info("audio_chat: transcript=%r user=%s", transcript[:80], uid)

    # Получаем ответ через стандартный chat pipeline
    import httpx as _httpx
    from app.core.config import get_settings

    settings = get_settings()
    used_model = model or settings.LLM_MODEL

    try:
        async with _httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.MWS_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.MWS_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": used_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Ты голосовой ассистент МТС. Отвечай кратко и по делу. "
                                "Пользователь общается голосом, поэтому отвечай разговорным языком "
                                "без Markdown-форматирования."
                            ),
                        },
                        {"role": "user", "content": transcript},
                    ],
                    "temperature": 0.7,
                    "stream": False,
                    "user": uid,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            reply = data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error("audio_chat: LLM call failed: %s", e)
        reply = f"Вы сказали: «{transcript}». К сожалению, не удалось получить ответ от ИИ: {str(e)[:100]}"

    return JSONResponse(
        content={
            "id": f"audiochat-{uuid.uuid4().hex[:12]}",
            "created": int(time.time()),
            "transcript": transcript,
            "reply": reply,
            "model": used_model,
        }
    )
