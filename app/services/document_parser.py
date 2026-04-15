"""
app/services/document_parser.py

Универсальный сервис-парсер для извлечения текста из различных источников:
PDF, DOCX, XLSX, CSV, .txt/.md и веб-ссылок.
Все данные конвертируются в чистый Markdown для передачи в контекст LLM.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import aiofiles
import fitz  # PyMuPDF
import httpx
import pandas as pd
from docx import Document

logger = logging.getLogger("mts.parser")

# ── Константы ─────────────────────────────────────────────────────────────────

JINA_BASE_URL = "https://r.jina.ai/"
WEBPAGE_TIMEOUT = 10.0          # секунды
CSV_EXCEL_ROW_LIMIT = 300       # максимальное число строк из таблиц


# ── Парсеры ───────────────────────────────────────────────────────────────────


async def parse_webpage(url: str) -> str:
    """
    Асинхронно загружает веб-страницу через Jina Reader API и возвращает
    её содержимое в формате Markdown.

    Args:
        url: Полный URL страницы (включая схему http/https).

    Returns:
        Текст страницы в Markdown или строка с описанием ошибки.
    """
    jina_url = f"{JINA_BASE_URL}{url}"
    headers = {"X-Return-Format": "markdown"}

    try:
        async with httpx.AsyncClient(timeout=WEBPAGE_TIMEOUT) as client:
            response = await client.get(jina_url, headers=headers)
            response.raise_for_status()
            return response.text
    except httpx.TimeoutException as exc:
        error_msg = f"[Система: Таймаут при загрузке страницы '{url}': {exc}]"
        logger.error(error_msg)
        return error_msg
    except httpx.HTTPStatusError as exc:
        error_msg = (
            f"[Система: HTTP-ошибка {exc.response.status_code} "
            f"при загрузке страницы '{url}': {exc}]"
        )
        logger.error(error_msg)
        return error_msg
    except Exception as exc:
        error_msg = f"[Система: Не удалось загрузить страницу '{url}': {exc}]"
        logger.error(error_msg, exc_info=True)
        return error_msg


async def parse_pdf(file_path: str) -> str:
    """
    Асинхронно извлекает текст из PDF-документа с постраничными метками.

    Args:
        file_path: Абсолютный путь к PDF-файлу.

    Returns:
        Текст документа в Markdown с разметкой страниц или строка с ошибкой.
    """
    def _extract() -> str:
        doc = fitz.open(file_path)
        parts: list[str] = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                parts.append(f"## [Страница {i + 1}]\n\n{text.strip()}")
        doc.close()
        return "\n\n---\n\n".join(parts)

    try:
        return await asyncio.to_thread(_extract)
    except Exception as exc:
        error_msg = f"[Система: Не удалось разобрать PDF '{file_path}': {exc}]"
        logger.error(error_msg, exc_info=True)
        return error_msg


async def parse_excel_csv(file_path: str) -> str:
    """
    Асинхронно читает таблицу XLSX или CSV и возвращает первые
    ``CSV_EXCEL_ROW_LIMIT`` строк в виде Markdown-таблицы.

    Args:
        file_path: Абсолютный путь к файлу .xlsx или .csv.

    Returns:
        Markdown-таблица или строка с описанием ошибки.
    """
    _, ext = os.path.splitext(file_path)

    def _read_table() -> str:
        if ext.lower() == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        return df.head(CSV_EXCEL_ROW_LIMIT).to_markdown(index=False)

    try:
        return await asyncio.to_thread(_read_table)
    except Exception as exc:
        error_msg = (
            f"[Система: Не удалось разобрать таблицу '{file_path}': {exc}]"
        )
        logger.error(error_msg, exc_info=True)
        return error_msg


async def parse_docx(file_path: str) -> str:
    """
    Асинхронно извлекает текст из документа DOCX, объединяя абзацы.

    Args:
        file_path: Абсолютный путь к файлу .docx.

    Returns:
        Текст документа или строка с описанием ошибки.
    """
    def _extract() -> str:
        doc = Document(file_path)
        return "\n".join(
            para.text for para in doc.paragraphs if para.text.strip()
        )

    try:
        return await asyncio.to_thread(_extract)
    except Exception as exc:
        filename = os.path.basename(file_path)
        error_msg = f"[ОШИБКА ПАРСИНГА DOCX: Не удалось прочитать файл {filename}]"
        logger.error(f"Ошибка при парсинге DOCX '{file_path}': {exc}", exc_info=True)
        return error_msg


async def _read_plain_text(file_path: str) -> str:
    """Читает обычный текстовый файл (.txt, .md) асинхронно."""
    try:
        async with aiofiles.open(file_path, encoding="utf-8") as f:
            return await f.read()
    except Exception as exc:
        error_msg = f"[Система: Не удалось прочитать файл '{file_path}': {exc}]"
        logger.error(error_msg, exc_info=True)
        return error_msg


# ── Единая точка входа ────────────────────────────────────────────────────────

# Соответствие расширений → обработчики
_EXT_HANDLERS: dict[str, object] = {
    ".pdf":  parse_pdf,
    ".docx": parse_docx,
    ".xlsx": parse_excel_csv,
    ".csv":  parse_excel_csv,
    ".txt":  _read_plain_text,
    ".md":   _read_plain_text,
}

_UNSUPPORTED_MSG = "[Система: Формат файла не поддерживается для извлечения текста]"


async def extract_text_from_source(source: str) -> str:
    """
    Универсальная точка входа: определяет тип источника и вызывает
    соответствующий парсер.

    Логика маршрутизации:
    - URL (http:// / https://) → parse_webpage
    - .pdf                    → parse_pdf
    - .docx                   → parse_docx
    - .xlsx / .csv            → parse_excel_csv
    - .txt / .md              → чтение как plain-text
    - всё остальное           → сообщение о неподдерживаемом формате

    Args:
        source: URL или абсолютный (либо относительный) путь к файлу.

    Returns:
        Извлечённый текст в Markdown или информационная строка об ошибке.
    """
    if source.startswith("http://") or source.startswith("https://"):
        logger.info("Парсинг веб-страницы: %s", source)
        return await parse_webpage(source)

    _, ext = os.path.splitext(source)
    handler = _EXT_HANDLERS.get(ext.lower())

    if handler is None:
        logger.warning(
            "Неподдерживаемый формат файла '%s' (расширение: '%s')", source, ext
        )
        return _UNSUPPORTED_MSG

    logger.info("Парсинг файла '%s' (тип: %s)", source, ext.lower())
    return await handler(source)  # type: ignore[operator]
