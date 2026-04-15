import asyncio
import logging
import re
import uuid
import time
import json
from typing import Any, AsyncGenerator

import httpx
from fastapi import APIRouter, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.config import get_settings, MODE_CHAT, MODE_AUTO, MODE_SEARCH, MODE_RESEARCH
from app.core.classifier import classify_auto_request, CONFIRM_WORDS
from app.core.prompts import search_contextualization_prompt, system_prompt
from app.memory.qdrant_store import memory_store, extract_and_save_facts, build_memory_context
from app.services.mws_client import generate_image

settings = get_settings()
router = APIRouter()
logger = logging.getLogger("mts.api")

class ChatMessage(BaseModel):
    role: str
    content: str | list[Any]

class ChatCompletionRequest(BaseModel):
    model: str = "llama-3.1-8b-instruct"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float = 0.7
    user: str | None = None

def _extract_research_depth(text: str) -> int | None:
    match = re.search(r"\[глубина[:\s]*(\d+)\]", text, re.IGNORECASE)
    if match:
        return min(max(1, int(match.group(1))), 10)
    return None

def is_utility_request(messages: list[ChatMessage]) -> bool:
    if not messages: return False
    last_content = str(messages[-1].content).lower()
    utility_patterns = ["generate 1-3 broad tags", "title for this conversation", "summary", "task:", "generate a short title", "create tags"]
    return any(p in last_content for p in utility_patterns)

def _is_pure_confirmation(text: str) -> bool:
    words = text.lower().strip().split()
    if len(words) > 3:
        return False
    return all(w in CONFIRM_WORDS for w in words)

def _stringify_message_content(content: Any) -> str:
    if content is None: return ""
    if isinstance(content, str): return content
    if isinstance(content, list):
        return "".join([str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in content])
    return str(content)

def trim_history(messages: list[dict], limit: int = 15) -> list[dict]:
    if len(messages) <= limit: return messages
    sys_msg = next((m for m in messages if m["role"] == "system"), None)
    recent = messages[-limit:]
    if sys_msg and sys_msg not in recent: return [sys_msg] + recent
    return recent

def inject_system_prompt(messages: list[dict], memory_context: str) -> list[dict]:
    sys_content = system_prompt.get_prompt(memory_context=memory_context)

    if messages and messages[0]["role"] == "system":
        original_sys = messages[0]["content"].replace("Avoid mentioning that you obtained the information from the context.", "").replace("Avoid mentioning that you obtained the information from the context", "")
        messages[0]["content"] = original_sys + "\n\n" + sys_content
        return messages
    
    return [{"role": "system", "content": sys_content}] + messages

async def contextualize_query(messages: list[dict], current_query: str) -> str:
    if len(messages) < 2:
        return current_query

    history_msgs = [m for m in messages if m.get("role") in ("user", "assistant")][-4:]
    if not history_msgs:
        return current_query
        
    history = "\n".join([f"{m['role']}: {m['content']}" for m in history_msgs])
    prompt = search_contextualization_prompt.get_prompt(query=current_query, conversation_history=history)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{settings.MWS_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {settings.MWS_API_KEY}"},
                json={
                    "model": settings.LLM_MODEL_ALT, 
                    "messages": [{"role": "user", "content": prompt}], 
                    "temperature": 0,
                    "max_tokens": 100,
                }
            )
            if resp.status_code == 200:
                data = resp.json()
                if "choices" in data and len(data["choices"]) > 0:
                    raw = data["choices"][0]["message"]["content"]
                    return search_contextualization_prompt.validate_output(raw) or current_query
            return current_query
    except Exception as e:
        logger.error(f"Contextualize query error: {e}")
        return current_query

async def stream_generator(messages: list[dict], model: str, user_id: str) -> AsyncGenerator[str, None]:
    headers = {"Authorization": f"Bearer {settings.MWS_API_KEY}"}
    payload = {"model": model, "messages": messages, "stream": True, "temperature": 0.7, "user": user_id}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=60.0)) as client:
            async with client.stream("POST", f"{settings.MWS_BASE_URL}/chat/completions", json=payload, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line.strip() != "data: [DONE]":
                        yield f"{line}\n\n"
                yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_chunk = {"id": f"err-{uuid.uuid4().hex[:6]}", "object": "chat.completion.chunk", "choices": [{"delta": {"content": f" Ошибка: {str(e)}" }}]}
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

def _quick_response(content: str, model: str = "") -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
        "model": model or settings.LLM_MODEL, "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]
    }

async def _handle_chat(request: ChatCompletionRequest, messages_dict: list[dict], user_id: str, model: str):
    logger.info("Chat routing: model=%s msgs=%s stream=%s", model, len(messages_dict), request.stream)
    if request.stream:
        return StreamingResponse(
            stream_generator(messages_dict, model, user_id),
            media_type="text/event-stream",
            headers={"Content-Type": "text/event-stream; charset=utf-8"},
        )
    headers = {"Authorization": f"Bearer {settings.MWS_API_KEY}"}
    payload = {"model": model, "messages": messages_dict, "stream": False, "temperature": request.temperature, "user": user_id}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{settings.MWS_BASE_URL}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()

async def _handle_search_chat(request: ChatCompletionRequest, messages_dict: list[dict], user_id: str, memory_context: str):
    from app.graph.search_chat import search_chat_stream
    return StreamingResponse(
        search_chat_stream(messages=messages_dict, user_id=user_id, model=settings.LLM_MODEL, memory_context=memory_context),
        media_type="text/event-stream", headers={"Content-Type": "text/event-stream; charset=utf-8"},
    )

async def _handle_deep_research(request: ChatCompletionRequest, messages_dict: list[dict], user_id: str, last_user_msg: str, memory_context: str):
    from app.graph.deep_research import deep_research_stream
    depth = _extract_research_depth(last_user_msg)
    clean_query = re.sub(r"\[глубина[:\s]*\d+\]", "", last_user_msg).strip()
    return StreamingResponse(
        deep_research_stream(query=clean_query or last_user_msg, user_id=user_id, messages=messages_dict, model=settings.LLM_MODEL, memory_context=memory_context, max_questions=depth),
        media_type="text/event-stream", headers={"Content-Type": "text/event-stream; charset=utf-8"},
    )

async def _handle_vlm(request: ChatCompletionRequest, user_id: str, memory_context: str):
    vlm_model = settings.VLM_MODEL
    sys_content = "Ты визуальный ассистент. Опиши изображение подробно."
    if memory_context: sys_content += f"\n\nКонтекст:\n{memory_context}"
    
    messages_raw: list[dict] = [{"role": "system", "content": sys_content}]
    for msg in request.messages:
        if msg.role == "system": continue
        if isinstance(msg.content, list):
            messages_raw.append({"role": msg.role, "content": msg.content})
        else:
            messages_raw.append({"role": msg.role, "content": str(msg.content or "")})
            
    if request.stream:
        return StreamingResponse(stream_generator(messages_raw, vlm_model, user_id), media_type="text/event-stream")
    return _quick_response("VLM Image process skipped for non-stream", vlm_model)

async def _handle_image(last_user_msg: str, model: str):
    try:
        image_url = await generate_image(last_user_msg)
        return _quick_response(f"![Сгенерированное изображение]({image_url})", settings.IMAGE_MODEL)
    except Exception as e:
        return _quick_response(f"Ошибка генерации картинки: {e}", model)

async def _handle_presentation(last_user_msg: str, user_id: str):
    from app.api.pptx_routes import generate_presentation, PresentationRequest
    slides_match = re.search(r"(\d+)\s*(слайд|slide)", last_user_msg, re.IGNORECASE)
    slides_count = max(2, min(int(slides_match.group(1)) if slides_match else 5, 12))
    try:
        resp = await generate_presentation(PresentationRequest(topic=last_user_msg, slides_count=slides_count, style="corporate", language="ru", user_id=user_id))
        data = json.loads(resp.body)
        if "error" in data: return _quick_response(f"Ошибка PPTX: {data['error']}", settings.LLM_MODEL)
        reply = f"✅ **Презентация готова!** ({data['slide_count']} слайдов)\n\n{data.get('preview_markdown', '')}\n\n---\n📥 [**Скачать PPTX**](http://localhost:8000{data['download_url']})"
        return _quick_response(reply, settings.LLM_MODEL)
    except Exception as e:
        return _quick_response(f"Ошибка PPTX: {e}", settings.LLM_MODEL)

async def _handle_website(last_user_msg: str, user_id: str):
    from app.api.website_routes import generate_website, WebsiteRequest
    try:
        resp = await generate_website(WebsiteRequest(description=last_user_msg, user_id=user_id))
        data = json.loads(resp.body)
        if not data.get("success"):
            return _quick_response(f"Ошибка генерации сайта: {data.get('error')}", settings.LLM_MODEL)
        reply = (
            f"✅ **Веб-сайт успешно сгенерирован!**\n\n"
            f"**Проект:** {data['project_name']}\n"
            f"**Описание:** {data['description']}\n\n"
            f"📦 **Состав архива:**\n{data['files_list']}\n\n"
            f"---\n"
            f"📥 [**Скачать ZIP-архив сайта**](http://localhost:8000{data['download_url']})"
        )
        return _quick_response(reply, settings.LLM_MODEL)
    except Exception as e:
        logger.exception("Website handling failed")
        return _quick_response(f"Ошибка генерации сайта: {e}", settings.LLM_MODEL)

@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: Request, bg_tasks: BackgroundTasks):
    user_id = str(http_request.headers.get("x-user-id") or http_request.headers.get("x-openwebui-user-id") or request.user or "openwebui-user")
    mode = request.model

    if is_utility_request(request.messages):
        msgs = [{"role": msg.role, "content": _stringify_message_content(msg.content)} for msg in request.messages]
        return StreamingResponse(stream_generator(msgs, settings.LLM_MODEL_ALT, user_id), media_type="text/event-stream")

    last_user_msg = next((_stringify_message_content(m.content) for m in reversed(request.messages) if m.role == "user"), "")
    
    if last_user_msg and len(last_user_msg.split()) >= 3:
        bg_tasks.add_task(memory_store.add_memory, last_user_msg, user_id)
        bg_tasks.add_task(extract_and_save_facts, last_user_msg, user_id)

    memory_context = await build_memory_context(last_user_msg, user_id)
    messages_dict = [{"role": msg.role, "content": _stringify_message_content(msg.content)} for msg in request.messages]
    messages_dict = trim_history(messages_dict)
    messages_dict = inject_system_prompt(messages_dict, memory_context)

    if mode == MODE_CHAT:
        return await _handle_chat(request, messages_dict, user_id, settings.LLM_MODEL)
    
    if mode == MODE_SEARCH:
        is_confirm = _is_pure_confirmation(last_user_msg)
        contextualized = last_user_msg if is_confirm else await contextualize_query(messages_dict, last_user_msg)
        if contextualized != last_user_msg:
            messages_dict[-1]["content"] = contextualized
        return await _handle_search_chat(request, messages_dict, user_id, memory_context)
        
    if mode == MODE_RESEARCH:
        is_confirm = _is_pure_confirmation(last_user_msg)
        contextualized = last_user_msg if is_confirm else await contextualize_query(messages_dict, last_user_msg)
        return await _handle_deep_research(request, messages_dict, user_id, contextualized, memory_context)

    if mode == MODE_AUTO:
        has_image = any(isinstance(msg.content, list) and any(isinstance(b, dict) and b.get("type") == "image_url" for b in msg.content) for msg in request.messages if msg.role == "user")
        if has_image: return await _handle_vlm(request, user_id, memory_context)

        classification = await classify_auto_request(last_user_msg)
        
        if classification == "image": return await _handle_image(last_user_msg, request.model)
        if classification == "presentation": return await _handle_presentation(last_user_msg, user_id)
        if classification == "website": return await _handle_website(last_user_msg, user_id)
        if classification == "data":
            from app.graph.agent import agent_stream
            return StreamingResponse(
                agent_stream(messages_dict, user_id, MODE_AUTO, memory_context),
                media_type="text/event-stream", headers={"Content-Type": "text/event-stream; charset=utf-8"}
            )
        
        if classification == "research":
            is_confirm = _is_pure_confirmation(last_user_msg)
            if is_confirm:
                history_text = " ".join(m.get("content", "") for m in messages_dict if m.get("role") == "assistant")
                has_research_plan = any(marker in history_text for marker in ["Я подготовил план исследования", "План исследования:", "Бриф исследования"])
                if not has_research_plan:
                    return await _handle_chat(request, messages_dict, user_id, settings.LLM_MODEL)
            contextualized = last_user_msg if is_confirm else await contextualize_query(messages_dict, last_user_msg)
            return await _handle_deep_research(request, messages_dict, user_id, contextualized, memory_context)
            
        if classification in ("search", "scrape"):
            is_confirm = _is_pure_confirmation(last_user_msg)
            contextualized = last_user_msg if is_confirm else await contextualize_query(messages_dict, last_user_msg)
            if contextualized != last_user_msg:
                messages_dict[-1]["content"] = contextualized
            return await _handle_search_chat(request, messages_dict, user_id, memory_context)
        
        return await _handle_chat(request, messages_dict, user_id, settings.LLM_MODEL)

    return await _handle_chat(request, messages_dict, user_id, mode)

@router.get("/models")
async def list_models():
    now = int(time.time())
    virtual_models = [
        {"id": MODE_AUTO, "object": "model", "created": now, "owned_by": "MTS Hackathon"},
        {"id": MODE_CHAT, "object": "model", "created": now, "owned_by": "MTS Hackathon"},
        {"id": MODE_SEARCH, "object": "model", "created": now, "owned_by": "MTS Hackathon"},
        {"id": MODE_RESEARCH, "object": "model", "created": now, "owned_by": "MTS Hackathon"},
    ]
    return {"object": "list", "data": virtual_models}

class EmbeddingsRequest(BaseModel):
    input: str | list[str]
    model: str = "qwen3-embedding-8b"
    user: str | None = None

@router.post("/embeddings")
async def embeddings(request: EmbeddingsRequest):
    if not request.input:
        return {"object": "list", "data": [], "model": request.model, "usage": {"prompt_tokens": 0, "total_tokens": 0}}
        
    headers = {"Authorization": f"Bearer {settings.MWS_API_KEY}"}
    payload = request.model_dump(exclude_none=True)
    payload["model"] = settings.EMBEDDING_MODEL
    
    if "user" in payload and payload["user"] is None:
        del payload["user"]
        
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{settings.MWS_BASE_URL}/embeddings", json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error(f"Embeddings Error: {e}")
        return {"object": "list", "data": [], "model": request.model, "error": str(e)}