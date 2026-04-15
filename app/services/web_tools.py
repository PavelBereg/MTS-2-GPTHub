"""
MCP-клиент для подключения к MCP-серверу веб-инструментов.

Управляет жизненным циклом MCP-подключения:
  - start_mcp()    — запускает MCP-сервер как subprocess, получает tools
  - stop_mcp()     — останавливает сервер
  - get_tools()    — возвращает список LangChain-совместимых tools
  - smart_search() — поиск через MCP (с fallback на прямой DDG)
  - smart_scrape() — парсинг через MCP (с fallback на прямой httpx)

Pydantic-модели:
  - SearchResult   — один результат поиска (title, url, snippet, domain, verified)
  - SearchResponse — полный ответ поиска (query, results, year)
  - ScrapeResponse — результат парсинга (url, title, content, domain, success)
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field

logger = logging.getLogger("mts.web_tools")


# Pydantic-модели для унификации данных


class SearchResult(BaseModel):
    """Один результат поиска."""

    title: str = ""
    url: str = ""
    snippet: str = ""
    domain: str = ""
    verified: bool = False  # URL проверен HEAD-запросом


class SearchResponse(BaseModel):
    """Полный ответ поиска."""

    query: str
    results: list[SearchResult] = Field(default_factory=list)
    year: int = Field(default_factory=lambda: datetime.now().year)
    source: str = "duckduckgo"  # Источник: mcp / duckduckgo

    def to_context_string(self) -> str:
        """Форматирует для вставки в контекст LLM."""
        if not self.results:
            return f"По запросу '{self.query}' ничего не найдено."

        parts = [f"Результаты поиска по запросу '{self.query}' ({self.year} год):\n"]
        for i, r in enumerate(self.results, 1):
            status = "" if r.verified else ""
            parts.append(
                f"[{i}] {status} {r.title}\n"
                f"    {r.snippet}\n"
                f"    URL: {r.url}\n"
                f"    Источник: {r.domain}"
            )
        return "\n\n".join(parts)

    def get_urls(self) -> list[str]:
        """Возвращает список проверенных URL (или всех если нет проверенных)."""
        verified = [r.url for r in self.results if r.verified and r.url]
        if verified:
            return verified
        return [r.url for r in self.results if r.url]


class ScrapeResponse(BaseModel):
    """Результат парсинга страницы."""

    url: str
    title: str = ""
    content: str = ""
    domain: str = ""
    success: bool = False
    error: str = ""
    char_count: int = 0

    def to_context_string(self) -> str:
        """Форматирует для вставки в контекст LLM."""
        if not self.success:
            return f"️ Не удалось прочитать {self.url}: {self.error}"

        header = f" {self.title}\n {self.url}\n {self.domain}\n\n"
        return header + self.content


# Глобальное состояние MCP-клиента

_client = None
_tools: list = []
_started: bool = False


async def start_mcp() -> None:
    """
    Запускает MCP-сервер как subprocess и подключается к нему.
    Вызывается один раз при старте FastAPI (lifespan).
    """
    global _client, _tools, _started

    if _started:
        logger.warning("MCP client already started, skipping")
        return

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        logger.error(
            " langchain-mcp-adapters не установлен! "
            "pip install langchain-mcp-adapters"
        )
        _started = True
        return

    server_path = str(
        Path(__file__).resolve().parent.parent.parent / "mcp_server" / "server.py"
    )
    python_path = sys.executable

    logger.info("Starting MCP server: %s %s", python_path, server_path)

    try:
        # В новых версиях langchain-mcp-adapters может потребоваться async with
        _client = MultiServerMCPClient(
            {
                "web_tools": {
                    "command": python_path,
                    "args": [server_path],
                    "transport": "stdio",
                }
            }
        )
        
        # Попытка получить инструменты. Если клиент требует инициализации, 
        # он может сделать это внутри get_tools или мы можем вызвать initialize.
        try:
            _tools = await _client.get_tools()
        except Exception as tool_err:
            logger.warning("Direct get_tools failed, trying context manager: %s", tool_err)
            # Если прямой вызов не сработал, возможно сервер ещё не готов
            _tools = []

        _started = True
        tool_names = [t.name for t in _tools]
        logger.info("MCP client initialized. Tools: %s", tool_names)

    except Exception as e:
        logger.error("Failed to start MCP client: %s", e, exc_info=True)
        _client = None
        _tools = []
        _started = True


async def stop_mcp() -> None:
    """Останавливает MCP-клиент и subprocess."""
    global _client, _tools, _started

    if _client is not None:
        try:
            if hasattr(_client, "close"):
                await _client.close()
            elif hasattr(_client, "__aexit__"):
                await _client.__aexit__(None, None, None)
            
            logger.info("MCP client stopped")
        except Exception as e:
            logger.warning("Error stopping MCP client: %s", e)

    _client = None
    _tools = []
    _started = False


def get_tools() -> list:
    """Возвращает список LangChain tools из MCP-сервера."""
    return list(_tools)


def get_tool_by_name(name: str):
    """Возвращает конкретный tool по имени, или None."""
    for tool in _tools:
        if tool.name == name:
            return tool
    return None


def is_available() -> bool:
    """Проверяет, доступен ли MCP-клиент."""
    return _started and len(_tools) > 0


def _extract_domain(url: str) -> str:
    """Извлекает домен из URL."""
    try:
        return urlparse(url).netloc or url
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


# Fallback: прямые функции (если MCP недоступен)

# Современный User-Agent для обхода блокировок
MODERN_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

async def _direct_search(query: str, max_results: int = 5) -> str:
    """Прямой поиск через DuckDuckGo без MCP (fallback)."""
    from ddgs import DDGS

    current_year = datetime.now().year

    # Добавляем год только если запрос короткий и не содержит год/временной метки
    time_words = [str(current_year), str(current_year - 1), "сейчас", "сегодня", "текущий", "latest", "recent", "2024", "2025", "2026"]
    has_time_marker = any(w in query.lower() for w in time_words)
    # Добавляем год только для коротких поисковых запросов (не для длинных вопросов)
    if not has_time_marker and len(query.split()) <= 5:
        enhanced_query = f"{query} {current_year}"
    else:
        enhanced_query = query

    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(enhanced_query, max_results=max_results + 3)
            )
    except Exception as e:
        logger.warning("DDGS primary search failed: %s", e)
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
        except Exception as e2:
            return f"️ Ошибка поиска: {str(e2)}"

    if not results:
        return f"По запросу '{query}' ничего не найдено."

    # Фильтр мусорных доменов (словари, синонимы, калькуляторы, агрегаторы отзывов)
    _JUNK_DOMAINS = {
        "sinonim.org", "ru.hinative.com", "ru.m.wiktionary.org", "wiktionary.org",
        "tripadvisor.com", "tripadvisor.in", "tripadvisor.ru", "yelp.com",
        "calc-online.xyz", "calculeitor.com", "fractioncalculator.pro",
        "zhihu.com", "ithelp.ithome.com.tw", "blogs.bing.com", "www.bing.com",
        "dic.academic.ru", "sinonim.com", "synonyms.ru", "gramota.ru",
    }

    filtered = []
    for r in results:
        url = r.get("href") or r.get("link") or r.get("url") or ""
        domain = _extract_domain(url)
        if domain in _JUNK_DOMAINS or any(jd in domain for jd in ["wiktionary", "tripadvisor", "yelp", "calc"]):
            logger.debug("[SEARCH] Filtered junk domain: %s", domain)
            continue
        filtered.append(r)

    # Если после фильтрации ничего не осталось — берём исходные
    results = filtered if filtered else results

    formatted = []
    for i, r in enumerate(results, 1):
        # Поддержка разных версий duckduckgo-search
        url = r.get("href") or r.get("link") or r.get("url") or ""
        url = _clean_url(url)
        domain = _extract_domain(url)
        title = r.get("title", "Без заголовка")
        body = r.get("body", "") or r.get("snippet", "")
        
        formatted.append(
            f"[{i}] {title}\n"
            f"    {body}\n"
            f"    URL: {url}\n"
            f"    Источник: {domain}"
        )
    return (
        f"Результаты поиска (актуальные, {current_year}):\n\n"
        + "\n\n".join(formatted)
    )


async def _direct_scrape(url: str, max_chars: int = 8000) -> str:
    """Прямой парсинг страницы без MCP (fallback)."""
    import httpx
    from bs4 import BeautifulSoup

    if not url.startswith(("http://", "https://")):
        return f"️ Ошибка: URL должен начинаться с http:// или https:// (получено: {url})"

    try:
        async with httpx.AsyncClient(
            verify=False,
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": MODERN_UA},
        ) as client:
            response = await client.get(url)
            if response.status_code == 404:
                return f"️ Страница не найдена (404): {url}"
            if response.status_code >= 400:
                return f"️ HTTP ошибка {response.status_code}: {url}"
    except Exception as e:
        return f"️ Ошибка загрузки {url}: {str(e)[:200]}"

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "button"]):
            tag.extract()

        title = soup.title.string.strip() if soup.title and soup.title.string else "No Title"
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)

        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[... текст обрезан ...]"

        domain = _extract_domain(url)
        header = f" {title}\n {url}\n {domain}\n\n"
        return (header + text) if text.strip() else f"️ Страница {url} не содержит полезного текста."
    except Exception as e:
        return f"️ Ошибка парсинга {url}: {str(e)[:200]}"



# Smart-функции: MCP → fallback (возвращают строку)


async def smart_search(query: str, max_results: int = 5) -> str:
    """Поиск: MCP если доступен, иначе прямой DuckDuckGo."""
    if is_available():
        tool = get_tool_by_name("search_web")
        if tool:
            try:
                result = await tool.ainvoke(
                    {"query": query, "max_results": max_results}
                )
                return result if isinstance(result, str) else str(result)
            except Exception as e:
                logger.warning("MCP search_web failed, fallback: %s", e)

    return await _direct_search(query, max_results)


async def smart_scrape(url: str, max_chars: int = 8000) -> str:
    """Парсинг: MCP если доступен, иначе прямой httpx+BS4."""
    if is_available():
        tool = get_tool_by_name("scrape_url")
        if tool:
            try:
                result = await tool.ainvoke({"url": url, "max_chars": max_chars})
                return result if isinstance(result, str) else str(result)
            except Exception as e:
                logger.warning("MCP scrape_url failed, fallback: %s", e)

    return await _direct_scrape(url, max_chars)


# Structured-функции: возвращают Pydantic-модели


def _parse_search_results(raw: str, query: str) -> SearchResponse:
    """Парсит raw-строку с результатами поиска в SearchResponse."""
    results = []

    # Парсим блоки типа:
    # [1] Title
    #     Snippet
    #     URL: https://...
    #     Источник: domain.com
    # Используем более гибкий разделитель
    blocks = re.split(r"(?=\[\d+\])", raw)
    for block in blocks:
        if not block.strip():
            continue

        # Улучшенный поиск URL, не захватывающий закрывающие скобки и знаки препинания в конце
        url_match = re.search(r"URL:\s*(https?://[^\s\)\],]+)", block)
        # Исправлено: .? вместо []? для корректного матчинга иконки (✅/❓)
        title_match = re.match(r"\[\d+\]\s*.?\s*(.*?)(?:\n|$)", block)
        source_match = re.search(r"Источник:\s*(\S+)", block)

        if url_match:
            url = _clean_url(url_match.group(1).strip())
            # Определяем verified по наличию галочки
            verified = "✅" in block

            # Извлекаем snippet — строки после title, до URL
            snippet_lines = []
            lines = block.split("\n")
            title_text = title_match.group(1).strip() if title_match else ""
            
            for line in lines:
                line = line.strip()
                if not line or any(line.startswith(p) for p in ("URL:", "Источник:", "[")):
                    continue
                if title_text and title_text in line and len(line) < len(title_text) + 5:
                    continue
                snippet_lines.append(line)
            
            snippet = " ".join(snippet_lines)

            results.append(
                SearchResult(
                    title=title_text or "Без заголовка",
                    url=url,
                    snippet=snippet[:500],
                    domain=source_match.group(1) if source_match else _extract_domain(url),
                    verified=verified,
                )
            )

    return SearchResponse(
        query=query,
        results=results,
        source="mcp" if is_available() else "duckduckgo",
    )


async def smart_search_structured(
    query: str, max_results: int = 5
) -> SearchResponse:
    """Поиск с возвратом структурированных данных (Pydantic)."""
    raw = await smart_search(query, max_results)
    return _parse_search_results(raw, query)


async def smart_scrape_structured(
    url: str, max_chars: int = 8000
) -> ScrapeResponse:
    """Парсинг с возвратом структурированных данных (Pydantic)."""
    raw = await smart_scrape(url, max_chars)

    is_error = raw.startswith("️")

    if is_error:
        return ScrapeResponse(
            url=url,
            domain=_extract_domain(url),
            success=False,
            error=raw,
        )

    # Извлекаем title из формата " Title\n URL\n..."
    title = ""
    title_match = re.search(r"\s*(.*?)(?:\n|$)", raw)
    if title_match:
        title = title_match.group(1).strip()

    # Убираем заголовочные строки, оставляем контент
    content = raw
    for prefix in ("", "", "", "Содержимое"):
        lines = content.split("\n")
        content = "\n".join(
            line for line in lines if not line.strip().startswith(prefix)
        )
    content = content.strip()

    return ScrapeResponse(
        url=url,
        title=title,
        content=content,
        domain=_extract_domain(url),
        success=True,
        char_count=len(content),
    )
