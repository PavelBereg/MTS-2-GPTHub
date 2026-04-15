"""
Website Generator — эндпоинт для генерации веб-сайтов.

Маршруты:
  POST /v1/websites/generate   — создаёт архив с сайтом по описанию
  GET  /v1/websites/download/{file_id} — скачивает готовый архив
  GET  /v1/websites/list/{user_id}     — список сайтов пользователя

Дизайн:
  - Современные HTML/CSS/JS сайты
  - Адаптивный дизайн
  - Использует подходящую модель для генерации кода

Интеграция в чат:
  Если classify_intent() → 'website', вызывать _handle_website_generation()
  в routes.py, который делает HTTP POST на этот эндпоинт.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
import zipfile
from pathlib import Path
from string import Template
from typing import Optional

import httpx
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.services.mws_client import generate_image as mws_generate_image

logger = logging.getLogger("mts.website")

router = APIRouter()

# Временная директория для хранения архивов сайтов
WEBSITE_DIR = Path("/tmp/mts_websites")
WEBSITE_DIR.mkdir(parents=True, exist_ok=True)

# In-memory реестр файлов: file_id → Path
_file_registry: dict[str, dict] = {}


# Pydantic-модели


class WebsiteRequest(BaseModel):
    description: str = Field(..., description="Описание сайта", min_length=3, max_length=1000)
    style: str = Field(
        default="modern",
        description="Стиль: modern | minimal | corporate | creative",
    )
    language: str = Field(default="ru", description="Язык контента: ru | en")
    user_id: str = Field(default="anonymous", description="ID пользователя")


class WebsiteFile(BaseModel):
    filename: str
    content: str
    file_type: str  # html, css, js, png, jpg, etc.


class WebsiteImage(BaseModel):
    filename: str
    description: str
    alt_text: str = ""


class WebsiteData(BaseModel):
    project_name: str
    description: str
    files: list[WebsiteFile]
    images: list[WebsiteImage] = []


# Генерация структуры через LLM

_WEBSITE_GENERATION_PROMPT = """\
Создай полный веб-сайт на основе описания: «$description»

Требования:
- Стиль: $style
- Язык контента: $language
- Современный адаптивный дизайн
- HTML5, CSS3, чистый JavaScript (без фреймворков)
- Минимум 3 файлов: index.html, styles.css, script.js
- Сделай красивое расположение разделов: четкие визуальные блоки, большие отступы, карточки и аккуратная типографика
- Должен быть hero-разделы на основе описания: $description
- На фоне hero-раздела или ключевого блока используй изображение как фон с наложением затемнения
- Включи одну локальную фоновую картинку в поле "images" и используй ее в CSS через background-image: url("images/background.jpg") или аналогичный путь
- Пути к изображениям должны быть относительными, например "images/hero.png" или "images/background.jpg"
- Создай аккуратный адаптивный дизайн, кнопки CTA и визуальные акценты
- Сделай сайт рабочим: подключи styles.css и script.js в index.html, добавь meta charset и viewport
- Все кнопки написанные тобой ДОЛЖНЫ БЫТЬ РАБОЧИМИ
- Сделай форму обратной связи с полями имя, email, сообщение и кнопкой отправить
- Не используй lorem ipsum, напиши реальные тексты на основе темы сайта
- JS должен обрабатывать мобильное меню и отправку формы (alert или console.log)

Структура ответа — ТОЛЬКО валидный JSON:
{{
  "project_name": "название-проекта",
  "description": "краткое описание сайта",
  "files": [
    {{
      "filename": "index.html",
      "content": "<!DOCTYPE html>\\n<html lang='ru'>\\n<head>... полный HTML код ...</head>\\n<body>...</body>\\n</html>",
      "file_type": "html"
    }},
    {{
      "filename": "styles.css",
      "content": "/* Полный CSS код */\\nbody {{ margin: 0; font-family: Arial; }}\\n...",
      "file_type": "css"
    }},
    {{
      "filename": "script.js",
      "content": "// Полный JavaScript код\\nconsole.log('Hello World');\\n...",
      "file_type": "js"
    }}
  ],
  "images": [
    {{
      "filename": "images/hero.png",
      "description": "Красивая сцена кафе на берегу моря",
      "alt_text": "Главный баннер сайта"
    }}
  ]
}}

Генерируй полноценный, рабочий код сайта!

ОТВЕЧАЙ ТОЛЬКО JSON. НИКАКИХ ОБРАТНЫХ ТИРЕ, НИКАКИХ ОБЪЯСНЕНИЙ, НИКАКИХ ФРАЗ ПЕРЕД JSON. JSON ДОЛЖЕН НАЧИНАТЬСЯ С "{" И ЗАКАНЧИВАТЬСЯ С "}".
"""


def _extract_json_from_text(text: str) -> str:
    """Извлекает первый валидный JSON-объект из строки ответа."""
    text = text.strip()

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response")

    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue

    raise ValueError("No balanced JSON object found in model response")


def _ensure_html_assets(html: str) -> str:
    """Гарантирует рабочую структуру HTML и подключение CSS/JS."""
    lower = html.lower()
    if "<!doctype html>" not in lower:
        html = "<!DOCTYPE html>\n" + html
        lower = html.lower()

    if "<head>" in lower:
        head_close = re.search(r"</head>", html, flags=re.IGNORECASE)
        if head_close:
            insert_pos = head_close.start()
            insert_snippets = []
            if "<meta charset" not in lower:
                insert_snippets.append('    <meta charset="UTF-8">')
            if "viewport" not in lower:
                insert_snippets.append('    <meta name="viewport" content="width=device-width, initial-scale=1">')
            if "href=\"styles.css\"" not in lower and "href='styles.css'" not in lower:
                insert_snippets.append('    <link rel="stylesheet" href="styles.css">')
            if insert_snippets:
                html = html[:insert_pos] + "\n" + "\n".join(insert_snippets) + "\n" + html[insert_pos:]
                lower = html.lower()

    if "<title>" not in lower and "<head>" in lower:
        title_pos = re.search(r"<head>", html, flags=re.IGNORECASE)
        if title_pos:
            insert_pos = title_pos.end()
            html = html[:insert_pos] + "\n    <title>Website</title>" + html[insert_pos:]
            lower = html.lower()

    if "src=\"script.js\"" not in lower and "src='script.js'" not in lower:
        body_close = re.search(r"</body>", html, flags=re.IGNORECASE)
        if body_close:
            html = html[:body_close.start()] + "    <script defer src=\"script.js\"></script>\n" + html[body_close.start():]

    return html


def _is_data_uri(value: str) -> bool:
    return isinstance(value, str) and value.startswith("data:image/")


def _save_data_uri_image(data_uri: str, path: Path) -> None:
    header, encoded = data_uri.split(",", 1)
    data = base64.b64decode(encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


async def _download_image(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        content = response.content
    with open(path, "wb") as f:
        f.write(content)


async def _generate_site_images(images: list[WebsiteImage]) -> list[WebsiteFile]:
    files: list[WebsiteFile] = []
    for image in images:
        try:
            result = await mws_generate_image(image.description, size="1024x1024")
            image_path = WEBSITE_DIR / image.filename
            if _is_data_uri(result):
                _save_data_uri_image(result, image_path)
            else:
                await _download_image(result, image_path)

            files.append(
                WebsiteFile(
                    filename=image.filename,
                    content="",
                    file_type=image.filename.split('.')[-1],
                )
            )
        except Exception as e:
            logger.warning("Image generation failed for %s: %s", image.filename, e)
    return files


async def generate_website_code(description: str, style: str, language: str) -> WebsiteData:
    """
    Генерирует код сайта через LLM.
    """
    settings = get_settings()
    prompt = Template(_WEBSITE_GENERATION_PROMPT).substitute(
        description=description,
        style=style,
        language=language,
    )

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, read=120.0)) as client:
            response = await client.post(
                f"{settings.MWS_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.MWS_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.LLM_MODEL,  # Используем топовую модель для генерации кода
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 8000,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()

        raw_content = data["choices"][0]["message"]["content"].strip()

        # Очищаем от <think> тегов если есть
        raw_content = re.sub(r"<think>[\s\S]*?</think>", "", raw_content).strip()

        # Извлекаем JSON из ответа, если модель вставила префикс или пояснения
        json_content = _extract_json_from_text(raw_content)

        website_data = WebsiteData.model_validate_json(json_content)

        # Убедимся, что HTML-файлы подключают CSS и JS и содержат корректный head.
        for file_data in website_data.files:
            if file_data.file_type.lower() == "html":
                file_data.content = _ensure_html_assets(file_data.content)

        if website_data.images:
            image_files = await _generate_site_images(website_data.images)
            website_data.files.extend(image_files)

        logger.info(
            " Website generated: %s with %s files and %s images",
            website_data.project_name,
            len(website_data.files),
            len(website_data.images),
        )
        return website_data

    except Exception as e:
        logger.error(f" Website generation failed: {e}")
        raise


async def create_website_archive(website_data: WebsiteData, user_id: str) -> str:
    """
    Создаёт ZIP-архив с файлами сайта.
    Возвращает file_id для скачивания.
    """
    file_id = str(uuid.uuid4())
    archive_path = WEBSITE_DIR / f"{file_id}.zip"

    try:
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_data in website_data.files:
                delete_temp = False
                if file_data.content:
                    temp_path = WEBSITE_DIR / f"temp_{file_id}_{file_data.filename}"
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(temp_path, 'w', encoding='utf-8') as f:
                        f.write(file_data.content)
                    delete_temp = True
                else:
                    source_path = WEBSITE_DIR / file_data.filename
                    if source_path.exists():
                        temp_path = source_path
                    else:
                        temp_path = WEBSITE_DIR / f"temp_{file_id}_{file_data.filename}"
                        temp_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(temp_path, 'w', encoding='utf-8') as f:
                            f.write(file_data.content)
                        delete_temp = True

                # Добавляем в архив
                zipf.write(temp_path, file_data.filename)

                # Удаляем временный файл, если он был временным
                if delete_temp and temp_path.exists():
                    temp_path.unlink()

        # Регистрируем файл
        _file_registry[file_id] = {
            "path": str(archive_path),
            "user_id": user_id,
            "project_name": website_data.project_name,
            "created_at": time.time(),
            "file_count": len(website_data.files),
        }

        logger.info(f" Website archive created: {file_id} ({website_data.project_name})")
        return file_id

    except Exception as e:
        logger.error(f" Archive creation failed: {e}")
        if archive_path.exists():
            archive_path.unlink()
        raise


# API эндпоинты


@router.post("/generate")
async def generate_website(request: WebsiteRequest) -> JSONResponse:
    """
    Генерирует веб-сайт и возвращает ссылку на скачивание архива.
    """
    logger.info(f" Generating website for user {request.user_id}: {request.description[:50]}...")

    try:
        # Генерируем код сайта
        website_data = await generate_website_code(
            request.description,
            request.style,
            request.language
        )

        # Создаём архив
        file_id = await create_website_archive(website_data, request.user_id)

        # Формируем preview (список файлов)
        files_list = "\n".join([f"- {f.filename} ({f.file_type})" for f in website_data.files])

        return JSONResponse({
            "success": True,
            "project_name": website_data.project_name,
            "description": website_data.description,
            "file_count": len(website_data.files),
            "files_list": files_list,
            "download_url": f"/v1/websites/download/{file_id}",
            "file_id": file_id,
        })

    except Exception as e:
        logger.exception("Website generation failed")
        return JSONResponse({
            "success": False,
            "error": f"Не удалось создать сайт: {str(e)[:200]}",
        }, status_code=500)


@router.get("/download/{file_id}")
async def download_website(file_id: str):
    """
    Скачивает архив с сайтом.
    """
    if file_id not in _file_registry:
        return JSONResponse({"error": "Файл не найден"}, status_code=404)

    file_info = _file_registry[file_id]
    archive_path = Path(file_info["path"])

    if not archive_path.exists():
        return JSONResponse({"error": "Файл не найден на диске"}, status_code=404)

    project_name = file_info.get("project_name", "website")
    filename = f"{project_name.replace(' ', '_')}.zip"

    return FileResponse(
        path=archive_path,
        filename=filename,
        media_type="application/zip",
    )


@router.get("/list/{user_id}")
async def list_user_websites(user_id: str) -> JSONResponse:
    """
    Возвращает список сайтов пользователя.
    """
    user_files = [
        {
            "file_id": fid,
            "project_name": info["project_name"],
            "created_at": info["created_at"],
            "file_count": info["file_count"],
            "download_url": f"/v1/websites/download/{fid}",
        }
        for fid, info in _file_registry.items()
        if info["user_id"] == user_id
    ]

    return JSONResponse({"websites": user_files})