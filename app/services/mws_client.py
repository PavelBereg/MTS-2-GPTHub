"""
Асинхронный клиент MWS API (OpenAI-совместимый).

Base URL и ключ задаются через константу и переменную окружения MWS_API_KEY.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger("mts.mws_client")

MWS_BASE_URL = "https://api.gpt.mws.ru/v1"

# Дефолт, если settings ещё не прочитаны; реальный приоритет — get_settings().IMAGE_MODEL
IMAGE_MODEL_FALLBACK = "qwen-image"
VISION_MODEL = "qwen2.5-vl-32b-instruct-awq"
TRANSCRIBE_MODEL = "whisper-turbo-local-preview"


class MWSAPIError(Exception):
    """Ошибка ответа MWS API или некорректного формата данных."""

    pass


def _api_key() -> str:
    key = os.getenv("MWS_API_KEY", "").strip()
    if not key or key == "CHANGE_ME":
        try:
            from app.core.config import get_settings

            s = get_settings().MWS_API_KEY.strip()
            if s and s != "CHANGE_ME":
                key = s
        except Exception:
            pass
    if not key or key == "CHANGE_ME":
        raise MWSAPIError(
            "Ключ MWS не задан: укажите MWS_API_KEY в окружении или в .env (не CHANGE_ME)."
        )
    return key


def _auth_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_api_key()}",
    }


def _error_message_from_response(response: httpx.Response) -> str:
    text_preview = (response.text or "")[:800]
    try:
        payload: Any = response.json()
    except Exception:
        return f"MWS API HTTP {response.status_code}: {text_preview or '(пустое тело)'}"

    err = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(err, dict):
        msg = err.get("message") or err.get("code") or str(err)
        return f"MWS API HTTP {response.status_code}: {msg}"
    if isinstance(err, str):
        return f"MWS API HTTP {response.status_code}: {err}"
    if isinstance(payload, dict) and "message" in payload:
        return f"MWS API HTTP {response.status_code}: {payload['message']}"
    return f"MWS API HTTP {response.status_code}: {text_preview or str(payload)}"


def _raise_for_status(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise MWSAPIError(_error_message_from_response(e.response)) from e


def _vision_image_url(image_base64: str) -> str:
    s = image_base64.strip()
    if s.startswith("data:"):
        return s
    # Сырой base64 без префикса — стандартный data URI для vision
    return f"data:image/png;base64,{s}"


async def generate_image(prompt_en: str, *, size: str | None = "1024x1024") -> str:
    """
    Генерация изображения через /images/generations.

    Возвращает публичный URL или data URI (base64), пригодный для markdown.
    Порядок моделей: settings.IMAGE_MODEL, затем запасные варианты при 404.
    """
    try:
        from app.core.config import get_settings

        primary_model = get_settings().IMAGE_MODEL.strip() or IMAGE_MODEL_FALLBACK
    except Exception:
        primary_model = IMAGE_MODEL_FALLBACK

    fallbacks = ["qwen-image", "sdxl-lightning-image"]
    try_order = [primary_model] + [m for m in fallbacks if m != primary_model]

    url = f"{MWS_BASE_URL}/images/generations"
    timeout = httpx.Timeout(30.0, read=300.0)

    last_response: httpx.Response | None = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for model_name in try_order:
            payload: dict[str, Any] = {
                "model": model_name,
                "prompt": prompt_en,
                "n": 1,
            }
            if size:
                payload["size"] = size
            logger.info(
                "images/generations: model=%s prompt_len=%s",
                model_name,
                len(prompt_en or ""),
            )
            response = await client.post(
                url,
                headers={**_auth_headers(), "Content-Type": "application/json"},
                json=payload,
            )
            last_response = response
            if response.status_code == 404:
                logger.warning(
                    "images/generations 404 for model=%s, trying next fallback",
                    model_name,
                )
                continue
            break
        else:
            response = last_response
    assert response is not None
    _raise_for_status(response)

    try:
        data = response.json()
    except Exception as exc:
        raise MWSAPIError("Ответ генерации изображения не является JSON.") from exc

    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        raise MWSAPIError("В ответе images/generations нет массива data или он пуст.")

    first = items[0]
    if not isinstance(first, dict):
        raise MWSAPIError("Элемент data[0] имеет неожиданный формат.")

    if first.get("url"):
        return str(first["url"])

    b64 = first.get("b64_json")
    if b64:
        # PNG по умолчанию для совместимости с OpenAI-style ответами
        return f"data:image/png;base64,{b64}"

    raise MWSAPIError(
        "В ответе нет ни url, ни b64_json для сгенерированного изображения."
    )


async def analyze_vision(image_base64: str, user_text: str) -> str:
    """
    Анализ изображения через /chat/completions (Vision).

    Возвращает текстовый ответ модели.
    """
    url = f"{MWS_BASE_URL}/chat/completions"
    image_url = _vision_image_url(image_base64)
    body = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
    }
    timeout = httpx.Timeout(30.0, read=180.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            url,
            headers={**_auth_headers(), "Content-Type": "application/json"},
            json=body,
        )
    _raise_for_status(response)

    try:
        data = response.json()
    except Exception as exc:
        raise MWSAPIError("Ответ chat/completions не является JSON.") from exc

    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        raise MWSAPIError("В ответе нет choices или список пуст.")

    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        raise MWSAPIError("Некорректная структура message в choices[0].")

    content = msg.get("content")
    if content is None:
        raise MWSAPIError("Поле message.content отсутствует в ответе.")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Некоторые провайдеры возвращают список частей
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                t = part.get("text")
                if isinstance(t, str):
                    parts.append(t)
        if parts:
            return "".join(parts)
    raise MWSAPIError("Не удалось извлечь текстовое содержимое из ответа vision.")


async def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Распознавание речи через /audio/transcriptions (multipart/form-data).

    Возвращает распознанный текст (поле text в JSON).
    """
    if not audio_bytes:
        raise MWSAPIError("Пустой аудиофайл (audio_bytes).")

    url = f"{MWS_BASE_URL}/audio/transcriptions"
    # Имя и MIME помогают бэкенду корректно распознать контейнер
    files = {
        "file": (
            "audio.webm",
            audio_bytes,
            "application/octet-stream",
        ),
    }
    data = {"model": TRANSCRIBE_MODEL}

    timeout = httpx.Timeout(30.0, read=300.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            url,
            headers=_auth_headers(),
            files=files,
            data=data,
        )
    _raise_for_status(response)

    try:
        payload = response.json()
    except Exception as exc:
        raise MWSAPIError("Ответ audio/transcriptions не является JSON.") from exc

    if isinstance(payload, dict) and "text" in payload:
        return str(payload["text"])

    raise MWSAPIError("В ответе транскрипции нет поля text.")