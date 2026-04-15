"""
LangGraph граф для режима « Поиск + Чат».

ReAct-агент с MCP-инструментами (search_web, scrape_url).
LLM решает, когда использовать инструменты.

Использование:
    from app.graph.search_chat import get_search_graph, search_chat_stream
    graph = get_search_graph()
    # Или для SSE:
    async for chunk in search_chat_stream(messages, user_id, ...):
        ...
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.core.config import get_settings
from app.graph.state import AgentState
from app.services import web_tools

settings = get_settings()
logger = logging.getLogger("mts.search_chat")

SEARCH_SYSTEM_PROMPT = """Ты — ИИ-ассистент МТС, который обязан давать конкретные ответы, а не быть поисковиком.

У тебя есть инструменты:
- search_web — поиск в интернете
- scrape_url — чтение содержимого страниц

КРИТИЧЕСКИЕ ПРАВИЛА:
1. ТВОЯ ЦЕЛЬ — ДАТЬ ПРЯМОЙ ОТВЕТ (САММАРИ). Никогда не отвечай просто списком ссылок.
2. Если пользователь спросил про погоду, курс валют или факты — найди цифру и назови её.
3. Если в сниппетах `search_web` нет конкретного ответа (например, температуры) — ОБЯЗАТЕЛЬНО используй `scrape_url` для захода на самый релевантный сайт.
4. Сначала дай текстовый ответ (Саммари), и только в самом конце прикрепи источники в формате Markdown [Заголовок](URL).
5. Отвечай ТОЛЬКО на русском языке.
6. Если данных нет нигде — так и скажи, не галлюцинируй.
7. Форматируй ответ красиво: используй жирный шрифт для ключевых фактов.

{memory_context}"""


# LLM-нода


async def search_llm_node(state: AgentState) -> dict:
    """LLM-нода с MCP-инструментами для поиска."""
    tools = web_tools.get_tools()
    memory = state.get("memory_context", "")
    memory_block = f"\nКонтекст из памяти:\n{memory}" if memory else ""

    system = SystemMessage(
        content=SEARCH_SYSTEM_PROMPT.replace("{memory_context}", memory_block)
    )
    messages = [system] + state["messages"]

    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(
        base_url=settings.MWS_BASE_URL,
        api_key=settings.MWS_API_KEY,
        model=state.get("model", settings.LLM_MODEL),
        temperature=0.5,
        streaming=False,
    )

    if tools:
        llm = llm.bind_tools(tools)

    try:
        response = await llm.ainvoke(messages)
    except Exception as e:
        logger.error("search_llm_node failed: %s", e, exc_info=True)
        response = AIMessage(content=f" Ошибка MWS API: {str(e)[:300]}")

    return {"messages": [response]}


# Сборка графа (lazy init — после запуска MCP)

_graph = None


def _build_search_graph():
    """Собирает ReAct граф для поиска + чата."""
    tools = web_tools.get_tools()

    graph = StateGraph(AgentState)
    graph.add_node("llm", search_llm_node)

    if tools:
        graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
        graph.set_entry_point("llm")
        graph.add_conditional_edges("llm", tools_condition)
        graph.add_edge("tools", "llm")
    else:
        # Без инструментов — просто LLM
        logger.warning("No MCP tools available, search_chat will run without tools")
        graph.set_entry_point("llm")
        graph.add_edge("llm", "__end__")

    return graph.compile()


def init_search_graph() -> None:
    """Инициализирует граф. Вызывать ПОСЛЕ start_mcp()."""
    global _graph
    _graph = _build_search_graph()
    tool_count = len(web_tools.get_tools())
    logger.info("Search chat graph initialized with %d tools", tool_count)


def get_search_graph():
    """Возвращает скомпилированный граф (lazy init)."""
    global _graph
    if _graph is None:
        init_search_graph()
    return _graph


# SSE Streaming wrapper


def _make_sse_chunk(content: str, model: str = "search-chat") -> str:
    """Создаёт SSE chunk в формате OpenAI."""
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _make_sse_finish(model: str = "search-chat") -> str:
    """Создаёт финальный SSE chunk с finish_reason=stop."""
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _extract_text(content) -> str:
    """Извлекает текст из content (строка или список блоков)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content) if content else ""


async def search_chat_stream(
    messages: list[dict],
    user_id: str,
    model: str = "",
    memory_context: str = "",
) -> AsyncGenerator[str, None]:
    """
    Запускает Search+Chat граф и стримит результат через SSE.

    Используется в routes.py:
        return StreamingResponse(
            search_chat_stream(messages, user_id, ...),
            media_type="text/event-stream",
        )
    """
    graph = get_search_graph()

    # Конвертируем dict-сообщения в LangChain Messages
    lc_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))

    initial_state: AgentState = {
        "messages": lc_messages,
        "user_id": user_id,
        "model": model or settings.LLM_MODEL,
        "memory_context": memory_context,
    }

    config = {
        "configurable": {
            "thread_id": f"search-{user_id}-{uuid.uuid4().hex[:8]}"
        }
    }

    sse_model = model or "search-chat"

    try:
        async for event in graph.astream(initial_state, config):
            for node_name, node_output in event.items():
                if node_name == "__end__":
                    continue

                out_messages = node_output.get("messages", [])
                for msg in out_messages:
                    if isinstance(msg, AIMessage):
                        text = _extract_text(msg.content)
                        if text:
                            yield _make_sse_chunk(text, sse_model)

    except Exception as e:
        logger.error("search_chat_stream error: %s", e, exc_info=True)
        yield _make_sse_chunk(f"\n\n Ошибка поиска: {str(e)[:300]}", sse_model)

    yield _make_sse_finish(sse_model)
    yield "data: [DONE]\n\n"
