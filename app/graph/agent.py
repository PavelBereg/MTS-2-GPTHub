"""
LangGraph агент — ядро пайплайна ИИ-ассистента.

AUTO-режим:
- обычный диалог → DeepSeek без tools (стабильный текст, без отказов «не могу вызвать функцию»);
- запрос картинки → прямой вызов MWS images/generations, без tool-calling;
- поиск / чтение URL → DeepSeek + только web_search и scrape.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from app.core.config import get_settings
from app.core.classifier import classify_intent
from app.graph.state import AgentState
from app.graph.tools import ALL_TOOLS, scrape_url, search_web
from app.services.mws_client import MWSAPIError, generate_image as mws_generate_image

settings = get_settings()
logger = logging.getLogger("mts.agent")

_TOOL_NAMES = {t.name for t in ALL_TOOLS}
_SEARCH_TOOLS = [search_web, scrape_url]
_SEARCH_TOOL_NAMES = {t.name for t in _SEARCH_TOOLS}

AUTO_MODEL_LABEL = "🌟 AUTO (Умный роутер МТС)"


def _strip_ui_garbage(text: str) -> str:
    """Убираем типичный мусор из промптов OpenWebUI / шаблонов и внутренние теги."""
    if not text:
        return ""
    s = text
    
    s = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL)
    s = re.sub(r'<think>.*', '', s, flags=re.DOTALL) 
    
    def _json_remover(match):
        content = match.group(0)
        try:
            val = json.loads(content)
            if isinstance(val, dict) and ("name" in val or "function" in val):
                return "" 
        except:
            pass
        return content
    
    s = re.sub(r'\{[^{}]*?"name"\s*:\s*".*?"[^{}]*?\}', _json_remover, s, flags=re.DOTALL)

    for pat in (
        r"</chat_history>",
        r"<chat_history>",
        r"</Chat_History>",
        r"<Chat_History>",
        r"</s>",
    ):
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


def _human_text(msg: HumanMessage) -> str:
    c = msg.content
    if isinstance(c, str):
        return _strip_ui_garbage(c)
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return _strip_ui_garbage("".join(parts))
    return _strip_ui_garbage(str(c))


def _last_user_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return _human_text(m)
    return ""


def _aimessage_text_content(message: AIMessage) -> str:
    c = message.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(c) if c is not None else ""


def _tool_calls_from_additional_kwargs(
    message: AIMessage, allowed_names: set[str]
) -> AIMessage:
    if getattr(message, "tool_calls", None):
        return message
    raw = (message.additional_kwargs or {}).get("tool_calls")
    if not isinstance(raw, list) or not raw:
        return message
    built: list[dict] = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if name not in allowed_names:
            continue
        args_raw = fn.get("arguments", "{}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        built.append(
            {
                "name": name,
                "args": args,
                "id": tc.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "type": "tool_call",
            }
        )
    if not built:
        return message
    return AIMessage(
        content=message.content,
        tool_calls=built,
        response_metadata=getattr(message, "response_metadata", None) or {},
        additional_kwargs=dict(message.additional_kwargs or {}),
    )


def _normalize_aimessage_tool_calls(
    message: AIMessage, allowed_names: set[str]
) -> AIMessage:
    if not allowed_names or getattr(message, "tool_calls", None):
        return message
    raw = _aimessage_text_content(message)
    if not raw.strip():
        return message

    decoder = json.JSONDecoder()
    extracted: list[dict] = []
    i = 0
    while i < len(raw):
        if raw[i] != "{":
            i += 1
            continue
        try:
            obj, end_rel = decoder.raw_decode(raw[i:])
        except json.JSONDecodeError:
            i += 1
            continue
        end = i + end_rel
        if isinstance(obj, dict):
            name = obj.get("name")
            if not name and obj.get("type") == "function":
                name = obj.get("function", {}).get("name") or obj.get("name")
            
            if name in allowed_names:
                params = obj.get("parameters") or obj.get("arguments") or obj.get("function", {}).get("arguments") or {}
                if isinstance(params, str):
                    try:
                        params = json.loads(params)
                    except json.JSONDecodeError:
                        params = {}
                
                extracted.append(
                    {
                        "name": name,
                        "args": params if isinstance(params, dict) else {},
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "tool_call",
                    }
                )
        i = end

    if not extracted and "execute_python" in allowed_names:
        import re
        match = re.search(r'```python\s*(.*?)\s*```', raw, re.DOTALL)
        if match:
            code = match.group(1).strip()
            extracted.append(
                {
                    "name": "execute_python",
                    "args": {"code": code},
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "tool_call",
                }
            )
            ranges.append(match.span())

    if not extracted:
        return message


    ranges = []
    i = 0
    while i < len(raw):
        if raw[i] != "{":
            i += 1
            continue
        try:
            obj, end_rel = decoder.raw_decode(raw[i:])
            end = i + end_rel
            if isinstance(obj, dict):
                name = obj.get("name")
                if not name and obj.get("type") == "function":
                    name = obj.get("function", {}).get("name")
                if name in allowed_names:
                    ranges.append((i, end))
            i = end
        except json.JSONDecodeError:
            i += 1

    clean_content = raw
    for start, end in reversed(ranges):
        clean_content = clean_content[:start] + clean_content[end:]

    clean_content = _strip_ui_garbage(clean_content)

    return AIMessage(
        content=clean_content,
        tool_calls=extracted,
        response_metadata=getattr(message, "response_metadata", None) or {},
    )


CHAT_SYSTEM_PROMPT = """Ты — полезный ассистент МТС. Отвечай по делу, на языке пользователя.
Не выдумывай отказы вида «не могу выполнить функцию»; обычные вопросы решай сам.

=== ИНСТРУКЦИИ ДЛЯ АНАЛИТИКИ И ВИЗУАЛИЗАЦИИ ===
1. ДАННЫЕ И ГРАФИКИ (Python Code Interpreter):
- ЕСЛИ ЗАДАЧА ТРЕБУЕТ ПОСТРОЕНИЯ ГРАФИКА ИЛИ EDA — ТЫ ОБЯЗАН ВЫЗВАТЬ `execute_python`.
- ЗАПРЕЩЕНО писать код в чат (```python). Только вызов инструмента.
- CRITICAL RULES FOR DATA ANALYSIS (EDA) AND PLOTTING:
  1. STOP TALKING, START CODING: If the user asks for analysis, EDA, or a chart, you are STRICTLY FORBIDDEN from just describing the data in text. You MUST actively write and execute Python code.
  2. SHOW, DON'T TELL: Do not list percentages or statistics in plain text if you were asked for a chart. Write a script to calculate and plot them.
  3. HOW TO LOAD DATA: You do not have physical files. Hardcode the data from context as a raw string and read using `io.StringIO()`.
  4. MANDATORY PLOTTING: You MUST save the plot to disk using `plt.savefig('static/downloads/filename.png')`.
  5. MANDATORY OUTPUT: Your final response MUST contain the absolute Markdown link: `![График](http://localhost:8000/static/downloads/filename.png)`.

Example of Expected Behavior:
User: Сделай EDA и нарисуй график по этим данным: [данные]
Assistant (Thinking): I must not just summarize this. I must write a Python script.
Assistant (Action): Calls `execute_python` with script using `io.StringIO` and `plt.savefig`.
```


2. СХЕМЫ, АЛГОРИТМЫ, АРХИТЕКТУРА (Mermaid.js):
Если пользователь просит нарисовать структуру базы данных, логическую схему, интеллект-карту, бизнес-процесс или таймлайн — НЕ используй Python и не вызывай инструмент.
Просто сгенерируй код на языке Mermaid.js и выдай его в текстовом ответе.
Обязательно оборачивай код строго в блок ```mermaid ... ```.
==============================================

{memory_context}"""

TOOLS_SEARCH_PROMPT = """Ты — ассистент МТС. У тебя есть доступ к поиску в интернете и чтению страниц по URL.
Используй search_web для актуальных фактов и новостей; scrape_url — если пользователь дал ссылку и просит прочитать страницу.
Отвечай на языке пользователя.

{memory_context}"""

SCRAPE_SYSTEM_PROMPT = """Ты — ассистент МТС. Тебе дали ссылку. 
Твоя единственная цель: прочитать содержимое через `scrape_url` и пересказать СУТЬ (сделать саммари).
НЕ используй поиск, если данные можно получить из ссылки.
Отвечай человеческим языком, выделяй главное.
В конце укажи ссылку как источник.

{memory_context}"""


async def llm_node(state: AgentState) -> dict:
    if not state.get("messages"):
        return {
            "messages": [
                AIMessage(
                    content="Пустой запрос: в сообщениях нет ни одной реплики для обработки."
                )
            ]
        }

    memory_ctx = state.get("memory_context", "")
    memory_block = f"\n\nКонтекст из памяти пользователя:\n{memory_ctx}" if memory_ctx else ""
    
    now = datetime.now()
    months = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"]
    date_str = f"{now.day} {months[now.month - 1]} {now.year}"
    
    query = _last_user_text(state["messages"])
    is_weather_query = bool(re.search(r"погод|температур|осадк|ветер", query.lower()))
    has_year = bool(re.search(r"\b(19\d{2}|20\d{2})\b", query))
    
    if is_weather_query:
        enhanced_query = f"{query} {date_str}"
    elif not has_year:
        enhanced_query = f"{query} {now.year}"
    else:
        enhanced_query = query

    model_request = state.get("model", settings.LLM_MODEL)
    last_user = enhanced_query
    
    logger.info(
        "llm_node: model_request=%r last_user_len=%s preview=%r",
        model_request,
        len(last_user),
        (last_user[:120] + "…") if len(last_user) > 120 else last_user,
    )

    if model_request != AUTO_MODEL_LABEL:
        system_msg = SystemMessage(
            content=TOOLS_SEARCH_PROMPT.replace("{memory_context}", memory_block).replace("{current_date}", date_str)
        )
        messages = [system_msg] + state["messages"]
        llm = ChatOpenAI(
            base_url=settings.MWS_BASE_URL,
            api_key=settings.MWS_API_KEY,
            model=model_request,
            temperature=0.7,
            streaming=False,
        ).bind_tools(ALL_TOOLS)
        try:
            response = await llm.ainvoke(messages)
        except Exception as e:
            logger.exception("MWS chat+tools failed model=%s", model_request)
            response = AIMessage(
                content=f" Ошибка MWS API (`{model_request}`): {str(e)[:300]}"
            )
        else:
            if isinstance(response, AIMessage):
                response = _tool_calls_from_additional_kwargs(response, _TOOL_NAMES)
                response = _normalize_aimessage_tool_calls(response, _TOOL_NAMES)
        return {"messages": [response]}

    messages_tail = state["messages"]

    intent = await classify_intent(last_user)
    
    # ── ПРОВЕРКА НА ПОВТОРНЫЙ ВХОД (ПОСЛЕ TOOL) ──
    from langchain_core.messages import ToolMessage
    has_tool_results = any(isinstance(m, ToolMessage) for m in messages_tail)
    has_python_result = any(isinstance(m, ToolMessage) and m.name == "execute_python" for m in messages_tail)

    if has_tool_results:
        logger.info(f"AUTO route=synthesis intent={intent} python={has_python_result}")
        
        if has_python_result:
            # Синтез для данных: ЗАПРЕЩАЕМ модели извиняться или предлагать повтор.
            system_msg = SystemMessage(
                content="ТЫ — АНАЛИТИК МТС. Анализ данных завершен. "
                        "Твоя задача: просто вывести результат (ссылку на график и текстовый вывод). "
                        "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО говорить 'Ой, что-то пошло не так' или предлагать 'давайте попробуем еще раз'. "
                        "Если в ответе инструмента есть ошибка — просто процитируй её как есть. "
                        "Если есть ссылка ![Chart](...) — ты ОБЯЗАН вывести её в финальном ответе."
            )
            # В блоке синтеза данных ИНСТРУМЕНТЫ БОЛЬШЕ НЕ НУЖНЫ (чтобы не зациклиться)
            llm = ChatOpenAI(
                base_url=settings.MWS_BASE_URL,
                api_key=settings.MWS_API_KEY,
                model=settings.CODER_MODEL,
                temperature=0.0,
                streaming=False,
            )
        else:
            # Синтез для поиска
            system_msg = SystemMessage(
                content=(
                    "ТЫ — АНАЛИТИК МТС. Выдай финальный ответ на основе поиска.\n"
                    f"Текущая дата: {datetime.now().strftime('%d.%m.%Y')}.\n"
                )
            )
            llm = ChatOpenAI(
                base_url=settings.MWS_BASE_URL,
                api_key=settings.MWS_API_KEY,
                model=settings.LLM_MODEL,
                temperature=0.2,
                streaming=False,
            ).bind_tools(_SEARCH_TOOLS)

        messages = [system_msg] + messages_tail
        try:
            response = await llm.ainvoke(messages)
            if isinstance(response, AIMessage) and not has_python_result:
                response = _tool_calls_from_additional_kwargs(response, _SEARCH_TOOL_NAMES)
                response = _normalize_aimessage_tool_calls(response, _SEARCH_TOOL_NAMES)
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")

    logger.info(f"AUTO route={intent} model_choice={settings.CODER_MODEL if intent == 'data' else settings.LLM_MODEL}")

    if intent == "image":
        logger.info("AUTO route=direct_image")
        try:
            src = await mws_generate_image(last_user, size="1024x1024")
            body = f"Готово:\n\n![Изображение]({src})"
        except MWSAPIError as e:
            logger.warning("direct_image MWS error: %s", e)
            body = f"Не удалось сгенерировать изображение: {e}"
        except Exception as e:
            logger.exception("direct_image unexpected")
            body = f"Ошибка генерации изображения: {str(e)[:200]}"
        return {"messages": [AIMessage(content=body)]}

    if intent in ("search", "scrape", "data"):
        if intent == "scrape":
            prompt_template = SCRAPE_SYSTEM_PROMPT
            active_tools = [scrape_url]
        elif intent == "data":
            prompt_template = CHAT_SYSTEM_PROMPT
            active_tools = ALL_TOOLS
        else:
            prompt_template = TOOLS_SEARCH_PROMPT
            active_tools = _SEARCH_TOOLS
            
        system_msg = SystemMessage(
            content=prompt_template.replace("{memory_context}", memory_block)
        )
        messages = [system_msg] + messages_tail
        active_tool_names = {t.name for t in active_tools}

        # КРИТИЧЕСКИ: используем CODER_MODEL для данных, иначе тулсы не вызываются
        model_name = settings.CODER_MODEL if intent == "data" else settings.LLM_MODEL

        llm = ChatOpenAI(
            base_url=settings.MWS_BASE_URL,
            api_key=settings.MWS_API_KEY,
            model=model_name,
            temperature=0.7,
            streaming=False,
        ).bind_tools(active_tools)
        
        try:
            response = await llm.ainvoke(messages)
        except Exception as e:
            logger.exception("MWS tool branch failed")
            response = AIMessage(
                content=f" Ошибка MWS API ({intent}): {str(e)[:300]}"
            )
        else:
            if isinstance(response, AIMessage):
                response = _tool_calls_from_additional_kwargs(
                    response, active_tool_names
                )
                response = _normalize_aimessage_tool_calls(
                    response, active_tool_names
                )
        return {"messages": [response]}

    last_raw = state["messages"][-1].content if state["messages"] else None
    has_image = isinstance(last_raw, list) and any(
        isinstance(b, dict) and b.get("type") == "image_url" for b in last_raw
    )
    if has_image:
        vlm = settings.VLM_MODEL
        logger.info("AUTO route=vision model=%s", vlm)
        system_msg = SystemMessage(
            content=(
                "Ты помогаешь описывать и обсуждать изображения пользователя. "
                "Отвечай на языке пользователя.\n"
                + memory_block
            )
        )
        messages = [system_msg] + messages_tail
        llm = ChatOpenAI(
            base_url=settings.MWS_BASE_URL,
            api_key=settings.MWS_API_KEY,
            model=vlm,
            temperature=0.7,
            streaming=False,
        )
        try:
            response = await llm.ainvoke(messages)
        except Exception as e:
            logger.exception("VLM failed")
            response = AIMessage(content=f"❌ Ошибка MWS API: {str(e)[:300]}")
        return {"messages": [response]}

    # 4) Обычный чат: без tools
    logger.info("AUTO route=plain_chat model=%s", settings.LLM_MODEL)
    system_msg = SystemMessage(
        content=CHAT_SYSTEM_PROMPT.replace("{memory_context}", memory_block)
    )
    messages = [system_msg] + messages_tail
    llm = ChatOpenAI(
        base_url=settings.MWS_BASE_URL,
        api_key=settings.MWS_API_KEY,
        model=settings.LLM_MODEL,
        temperature=0.7,
        streaming=False,
    ).bind_tools(ALL_TOOLS)
    try:
        response = await llm.ainvoke(messages)
    except Exception as e:
        logger.exception("plain_chat failed")
        response = AIMessage(content=f" Ошибка MWS API: {str(e)[:300]}")
    else:
        if isinstance(response, AIMessage):
            response = _tool_calls_from_additional_kwargs(response, _TOOL_NAMES)
            response = _normalize_aimessage_tool_calls(response, _TOOL_NAMES)
    return {"messages": [response]}


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("llm", llm_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS, handle_tool_errors=True))

    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", tools_condition)
    graph.add_edge("tools", "llm")

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


agent_executor = build_graph()


from typing import AsyncGenerator
import time

def _make_sse_chunk(content: str, model: str = "agent") -> str:
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

def _make_sse_finish(model: str = "agent") -> str:
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

async def agent_stream(
    messages: list[dict],
    user_id: str,
    model: str = "",
    memory_context: str = "",
) -> AsyncGenerator[str, None]:
    lc_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system": lc_messages.append(SystemMessage(content=content))
        elif role == "assistant": lc_messages.append(AIMessage(content=content))
        else: lc_messages.append(HumanMessage(content=content))

    model_req = model or settings.LLM_MODEL

    initial_state: AgentState = {
        "messages": lc_messages,
        "user_id": user_id,
        "model": model_req,
        "memory_context": memory_context,
    }
    config = {"configurable": {"thread_id": f"agent-{user_id}-{uuid.uuid4().hex[:8]}"}}
    sse_model = model_req

    try:
        async for event in agent_executor.astream(initial_state, config):
            for node_name, node_output in event.items():
                if node_name == "__end__": continue
                out_messages = node_output.get("messages", [])
                for msg in out_messages:
                    if isinstance(msg, AIMessage):
                        text = _aimessage_text_content(msg)
                        if text and text.strip():
                            yield _make_sse_chunk(text, sse_model)
    except Exception as e:
        logger.error("agent_stream error: %s", e, exc_info=True)
        yield _make_sse_chunk(f"\n\n Ошибка агента: {str(e)[:300]}", sse_model)

    yield _make_sse_finish(sse_model)
    yield "data: [DONE]\n\n"

