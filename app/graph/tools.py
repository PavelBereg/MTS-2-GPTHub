"""
LangChain Tools для MWS API. Прямая реализация без проблемных оберток.
"""

import logging
from typing import Annotated

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS  # Прямой импорт библиотеки поиска
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool

from app.services.mws_client import MWSAPIError, generate_image as mws_generate_image
from app.tools.python_sandbox import execute_python

logger = logging.getLogger("mts.tools")

def _clean_url(url: str) -> str:
    """Очищает URL от трекеров и кодирует для Markdown-безопасности."""
    from urllib.parse import urlparse, parse_qs, quote, urlunparse, unquote
    if not url: return ""
    
    # 1. Извлекаем реальный URL
    if "duckduckgo.com/l/" in url or "duckduckgo.com/y.js" in url:
        parsed_track = urlparse(url)
        query_track = parse_qs(parsed_track.query)
        if "uddg" in query_track:
            url = query_track["uddg"][0]
        elif "ad_url" in query_track:
            url = query_track["ad_url"][0]

    # 2. Безопасное кодирование
    try:
        url = unquote(url.strip())
        parsed = urlparse(url)
        safe_path = quote(parsed.path, safe="/%")
        safe_query = quote(parsed.query, safe="=&?/%")
        return urlunparse(parsed._replace(path=safe_path, query=safe_query))
    except Exception:
        return url

# ═══════════════════════════════════════════════════════════════════════════
# 1. Генерация изображений (url + b64_json через mws_client)
# LangGraph 0.6+ ToolNode требует ToolMessage, не str.
# ═══════════════════════════════════════════════════════════════════════════

@tool
async def generate_image_tool(
    prompt: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> ToolMessage:
    """
    Генерирует изображение по текстовому описанию (MWS images/generations).
    Всегда вызывай этот инструмент, если пользователь просит картинку / нарисовать / сгенерировать изображение.
    Не отвечай отказом без вызова инструмента — сначала вызови его с кратким описанием на русском или английском.
    """
    name = "generate_image_tool"
    logger.info("generate_image_tool: prompt_len=%s", len(prompt or ""))
    try:
        src = await mws_generate_image(prompt, size="1024x1024")
        text = f"![Сгенерированное изображение]({src})"
    except MWSAPIError as e:
        logger.warning("generate_image_tool MWS error: %s", e)
        text = f"⚠️ Ошибка генерации изображения (MWS): {e}"
    except Exception as e:
        logger.exception("generate_image_tool failed")
        text = f"⚠️ Ошибка генерации: {str(e)[:200]}"
    return ToolMessage(content=text, tool_call_id=tool_call_id, name=name)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Веб-поиск — ЗАДАЧА №7
# ═══════════════════════════════════════════════════════════════════════════

@tool
def search_web(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> ToolMessage:
    """
    Выполняет поиск в интернете через DuckDuckGo по ЛЮБОМУ непустому запросу.
    Вызывай обязательно, если нужны актуальные факты, новости, «что там в мире», погода, события, ссылки.
    Не отказывайся от вызова с формулировкой «неподходящий запрос» — сократи или переформулируй запрос (1–8 слов) и вызови инструмент.
    """
    name = "search_web"
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]

        if not results:
            text = f"По запросу '{query}' в интернете ничего не найдено."
        else:
            formatted_results = []
            for r in results:
                url = _clean_url(r.get("href", ""))
                formatted_results.append(
                    f"🔹 {r['title']}\n{r['body']}\nИсточник: {url}"
                )
            text = "Результаты поиска в сети:\n\n" + "\n\n".join(formatted_results)
    except Exception as e:
        text = f"⚠️ Ошибка поиска: Не удалось связаться с поисковым сервером. {str(e)}"
    return ToolMessage(content=text, tool_call_id=tool_call_id, name=name)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Веб-парсинг — ЗАДАЧА №8
# ═══════════════════════════════════════════════════════════════════════════

@tool
async def scrape_url(
    url: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> ToolMessage:
    """
    Читает текстовое содержимое веб-страницы по прямой ссылке.
    Используй, если пользователь прислал URL и просит проанализировать его содержимое.
    """
    name = "scrape_url"
    url = _clean_url(url.strip())
    if not url.startswith(("http://", "https://")):
        body = "Ошибка: URL должен начинаться с http:// или https://"
        return ToolMessage(content=body, tool_call_id=tool_call_id, name=name)

    try:
        async with httpx.AsyncClient(verify=False, timeout=15.0) as client:
            response = await client.get(url)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.extract()

        page_text = soup.get_text(separator=" ", strip=True)
        body = page_text[:5000] + ("..." if len(page_text) > 5000 else "")
    except Exception:
        body = f"⚠️ Ошибка парсинга: Не удалось прочитать сайт по ссылке {url}."
    return ToolMessage(content=body, tool_call_id=tool_call_id, name=name)


# ═══════════════════════════════════════════════════════════════════════════
# Список всех доступных инструментов
# ═══════════════════════════════════════════════════════════════════════════

ALL_TOOLS: list = [generate_image_tool, search_web, scrape_url, execute_python]
