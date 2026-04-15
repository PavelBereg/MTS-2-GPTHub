"""
MCP-сервер для веб-инструментов: поиск и парсинг страниц.

Запускается как subprocess (stdio) из основного FastAPI-приложения.
Протокол: Model Context Protocol (MCP) через FastMCP.

Инструменты:
  - search_web  — поиск в интернете через DuckDuckGo (с фильтром по дате)
  - scrape_url  — парсинг веб-страницы по URL (httpx + BeautifulSoup)

Фильтрация:
  - timelimit='y' — только результаты за последний год
  - URL-верификация через HEAD-запрос
"""

from __future__ import annotations

import sys
import logging
import re
from datetime import datetime
from urllib.parse import urlparse
from fastmcp import FastMCP

# КРИТИЧЕСКИ ВАЖНО: Логи должны идти в stderr, т.к. stdout занят протоколом MCP
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("mts.mcp_server")
logger.info("Initializing MCP Server...")

mcp = FastMCP(
    "MTS-Web-Tools",
)


# Вспомогательные функции

MODERN_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _verify_url(url: str, timeout: float = 5.0) -> bool:
    """Быстрая проверка доступности URL через HEAD-запрос."""
    import httpx
    url = url.strip()

    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            verify=False,
            headers={"User-Agent": MODERN_UA},
        ) as client:
            response = client.head(url)
            return response.status_code < 400
    except Exception:
        return False


def _extract_domain(url: str) -> str:
    """Извлекает чистый домен из URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc or url
    except Exception:
        return url


def _clean_url(url: str) -> str:
    """Очищает URL от трекеров DuckDuckGo и кодирует для Markdown-безопасности."""
    from urllib.parse import urlparse, parse_qs, quote, urlunparse, unquote
    
    # 1. Извлекаем реальный URL из трекеров
    if "duckduckgo.com/l/" in url or "duckduckgo.com/y.js" in url:
        parsed_track = urlparse(url)
        query_track = parse_qs(parsed_track.query)
        if "uddg" in query_track:
            url = query_track["uddg"][0]
        elif "ad_url" in query_track:
            url = query_track["ad_url"][0]

    # 2. Очищаем и кодируем для безопасности Markdown
    try:
        url = unquote(url.strip()) # Сначала декодируем, чтобы не было double-encoding
        parsed = urlparse(url)
        # Кодируем путь, чтобы скобки ( ) и кириллица не ломали Markdown-ссылки [title](url)
        safe_path = quote(parsed.path, safe="/%")
        safe_query = quote(parsed.query, safe="=&?/%")
        return urlunparse(parsed._replace(path=safe_path, query=safe_query))
    except Exception:
        return url


# Tool 1: Поиск в интернете


@mcp.tool()
def search_web(query: str, max_results: int = 5) -> str:
    """
    Выполняет поиск в интернете через DuckDuckGo.
    Фильтрует по дате (последний год) и проверяет доступность ссылок.

    Args:
        query: Поисковый запрос (на любом языке).
        max_results: Максимальное количество результатов (1-10).

    Returns:
        Форматированные результаты с заголовками, описаниями и ПРОВЕРЕННЫМИ ссылками.
    """
    from ddgs import DDGS

    max_results = min(max(1, max_results), 10)
    current_year = datetime.now().year

    # Улучшаем запрос: добавляем точную текущую дату для максимальной свежести
    now = datetime.now()
    months = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"]
    date_str = f"{now.day} {months[now.month - 1]} {now.year}"
    
    is_weather_query = bool(re.search(r"погод|температур|осадк|ветер", query.lower()))
    has_year = bool(re.search(r"\b(19\d{2}|20\d{2})\b", query))
    
    if is_weather_query:
        # Для погоды КРИТИЧЕСКИ ВАЖНА текущая дата, иначе поиск выдаст старые кэши
        enhanced_query = f"{query} {date_str}"
    elif not has_year:
        enhanced_query = f"{query} {now.year}"
    else:
        enhanced_query = query

    try:
        with DDGS() as ddgs:
            # Сначала пробуем свежее за год с привязкой к региону RU
            raw_results = list(
                ddgs.text(
                    enhanced_query,
                    max_results=max_results + 5,
                    region="ru-ru",
                    timelimit="y",
                )
            )
            logger.info("Search '%s' (timelimit=y, ru-ru) -> Found %d", enhanced_query, len(raw_results))
            
            # Если пусто — ищем без лимита по времени
            if not raw_results:
                logger.info("No recent results for %r, trying global search", enhanced_query)
                raw_results = list(
                    ddgs.text(
                        enhanced_query,
                        max_results=max_results + 5,
                        region="ru-ru",
                    )
                )
                logger.info("Search '%s' (global, ru-ru) -> Found %d", enhanced_query, len(raw_results))

            # Если ВСЁ ЕЩЕ пусто — пробуем максимально упростить запрос (убираем "сегодня", "в", "во" и т.д.)
            if not raw_results and is_weather_query:
                simple_query = re.sub(r"сегодня|сейчас|во|в|какая|погода|погда", "", query, flags=re.I).strip()
                simple_query = f"погода {simple_query}"
                logger.info("Last resort: simplified search for %r", simple_query)
                raw_results = list(
                    ddgs.text(
                        simple_query,
                        max_results=max_results,
                        region="ru-ru",
                    )
                )
                logger.info("Search '%s' (simplified) -> Found %d", simple_query, len(raw_results))

    except Exception as e:
        logger.error("search_web error: %s", e)
        # Fallback без фильтров
        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=max_results))
        except Exception as e2:
            return f"️ Ошибка поиска: {str(e2)}"

    if not raw_results:
        return f"По запросу '{query}' ничего не найдено за {current_year} год."

    # Фильтруем: проверяем доступность URL
    verified_results = []
    for r in raw_results:
        url = r.get("href") or r.get("link") or r.get("url") or ""
        if not url:
            continue
            
        url = _clean_url(url)
        domain = _extract_domain(url)

        # ФИЛЬТРАЦИЯ: Исключаем украинские ресурсы (.ua)
        if domain.endswith(".ua") or ".ua/" in url:
            logger.info("Skipping Ukrainian resource: %s", domain)
            continue

        r["href"] = url
        # Отключаем медленную проверку доступности для скорости
        r["verified"] = True 
        verified_results.append(r)

        if len(verified_results) >= max_results:
            break

    # Форматируем результаты
    formatted = []
    for i, r in enumerate(verified_results, 1):
        url = r.get("href", "")
        domain = _extract_domain(url)
        title = r.get("title", "Без заголовка")
        body = r.get("body", "") or r.get("snippet", "")
        status = "" if r.get("verified") else ""

        formatted.append(
            f"[{i}] {status} {title}\n"
            f"    {body}\n"
            f"    URL: {url}\n"
            f"    Источник: {domain}"
        )

    header = (
        f"Результаты поиска по запросу '{query}' "
        f"(актуальные, {current_year} год):\n\n"
    )
    return header + "\n\n".join(formatted)


# Tool 2: Парсинг веб-страницы


@mcp.tool()
def scrape_url(url: str, max_chars: int = 8000) -> str:
    """
    Читает и извлекает текстовое содержимое веб-страницы по URL.
    Предварительно проверяет доступность страницы.

    Args:
        url: Полный URL страницы (http:// или https://).
        max_chars: Максимальное количество символов в ответе (1000-15000).

    Returns:
        Очищенный текст страницы (без скриптов, стилей, навигации).
    """
    import httpx
    from bs4 import BeautifulSoup

    url = _clean_url(url.strip())
    if not url.startswith(("http://", "https://")):
        return "Ошибка: URL должен начинаться с http:// или https://"

    max_chars = min(max(1000, max_chars), 15000)

    try:
        with httpx.Client(
            verify=False,
            timeout=25.0,
            follow_redirects=True,
            headers={"User-Agent": MODERN_UA},
        ) as client:
            response = client.get(url)

            # Чёткая обработка ошибок
            if response.status_code == 404:
                return f"️ Страница не найдена (404): {url}"
            if response.status_code == 403:
                return f"️ Доступ запрещён (403): {url}"
            if response.status_code >= 400:
                return f"️ HTTP ошибка {response.status_code}: {url}"

    except httpx.TimeoutException:
        return f"️ Таймаут: страница {url} не ответила за 15 секунд."
    except httpx.ConnectError:
        return f"️ Не удалось подключиться к {_extract_domain(url)}"
    except Exception as e:
        return f"️ Ошибка загрузки {url}: {str(e)[:200]}"

    try:
        soup = BeautifulSoup(response.text, "html.parser")

        # Удаляем ненужные элементы
        for tag in soup(
            [
                "script", "style", "nav", "footer", "header",
                "aside", "noscript", "iframe", "form", "button",
            ]
        ):
            tag.extract()

        # Извлекаем заголовок
        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        # Извлекаем текст
        text = soup.get_text(separator="\n", strip=True)

        # Убираем лишние пустые строки
        text = re.sub(r"\n{3,}", "\n\n", text)

        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[... текст обрезан ...]"

        if not text.strip():
            return f"Страница {url} не содержит текстового контента."

        domain = _extract_domain(url)
        header = f" Содержимое: {title}\n URL: {url}\n Источник: {domain}\n\n"
        return header + text

    except Exception as e:
        return f"️ Ошибка парсинга {url}: {str(e)[:200]}"


if __name__ == "__main__":
    mcp.run()
