"""
Memory 2.0 — Fact Extraction (извлечение фактов о пользователе).

Вместо сохранения сырого текста чата анализирует диалог и вытаскивает
структурированные факты: имя, предпочтения, проекты, контекст работы.

Факты сохраняются в отдельную коллекцию Qdrant (user_facts) с метаданными.
Вызывается АСИНХРОННО в фоне — не блокирует ответ пользователю.

Использование:
    asyncio.create_task(
        extract_and_save_facts(user_message, assistant_reply, user_id)
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import NamedTuple

import httpx

from app.core.config import get_settings
from app.core.prompts import fact_extraction_prompt

logger = logging.getLogger("mts.fact_extractor")

# Промпт для извлечения фактов теперь в app/core/prompts.py


class ExtractedFact(NamedTuple):
    text: str
    fact_type: str


async def extract_facts(text: str) -> list[ExtractedFact]:
    """
    Вызывает llama-3.1-8b-instruct для извлечения фактов из текста.

    Args:
        text: Сообщение или фрагмент диалога.

    Returns:
        Список ExtractedFact. Пустой список если фактов нет или API недоступен.
    """
    if not text or len(text.strip()) < 10:
        return []

    settings = get_settings()
    prompt = fact_extraction_prompt.get_prompt(conversation_text=text[:800])

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, read=15.0)) as client:
            response = await client.post(
                f"{settings.MWS_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.MWS_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.LLM_MODEL_ALT,  # llama-3.1-8b — быстрый
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 256,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()

        raw = data["choices"][0]["message"]["content"].strip()

        facts_json = fact_extraction_prompt.validate_output(raw)
        results: list[ExtractedFact] = [
            ExtractedFact(text=item["fact"], fact_type=item["type"])
            for item in facts_json
        ]

        logger.info(
            "fact_extractor: extracted %d facts from %d chars",
            len(results),
            len(text),
        )
        return results

    except json.JSONDecodeError as e:
        logger.warning("fact_extractor: JSON parse error: %s", e)
        return []
    except httpx.TimeoutException:
        logger.warning("fact_extractor: timeout calling LLM")
        return []
    except Exception:
        logger.exception("fact_extractor: unexpected error")
        return []


async def extract_and_save_facts(
    user_message: str,
    user_id: str,
    assistant_reply: str = "",
) -> None:
    """
    Фоновая задача: извлекает факты и сохраняет их в Qdrant.

    Предназначена для вызова через asyncio.create_task() — не блокирует ответ.

    Args:
        user_message:    Последнее сообщение пользователя.
        user_id:         ID пользователя (для разграничения профилей).
        assistant_reply: Ответ ассистента (опционально, для контекста).
    """
    # Комбинируем контекст: факты могут быть и в ответе ассистента
    combined = user_message
    if assistant_reply:
        # Ищем упоминания пользователя в ответе ассистента
        combined = f"Пользователь: {user_message}\nАссистент: {assistant_reply[:300]}"

    facts = await extract_facts(combined)

    if not facts:
        logger.debug("fact_extractor: no facts to save for user %s", user_id)
        return

    # Импортируем здесь, чтобы избежать circular import
    from app.memory.qdrant_store import get_memory_store

    store = get_memory_store()

    import uuid
    from datetime import datetime, timezone
    
    # Сохраняем каждый факт как отдельный вектор (Long-term Store put)
    tasks = [
        store.put(
            namespace=("facts", user_id),
            key=str(uuid.uuid4()),
            value={
                "text": fact.text,
                "fact_type": fact.fact_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        for fact in facts
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    saved = sum(1 for r in results if not isinstance(r, Exception))
    failed = len(results) - saved
    if failed:
        logger.warning(
            "fact_extractor: saved %d/%d facts (failed: %d) for user %s",
            saved, len(results), failed, user_id,
        )
    else:
        logger.info(
            "fact_extractor: saved %d facts for user %s: %s",
            saved,
            user_id,
            [f.text[:40] for f in facts],
        )


async def build_memory_context(user_query: str, user_id: str) -> str:
    """
    Строит контекст памяти для передачи в промпт.

    Объединяет:
      1. Профиль пользователя (факты из user_facts)
      2. Семантически релевантные воспоминания из user_memory

    Returns:
        Форматированная строка для вставки в системный промпт.
    """
    from app.memory.qdrant_store import get_memory_store

    store = get_memory_store()

    # Запрашиваем параллельно
    profile_task = asyncio.create_task(store.get_user_profile(user_id, top_k=8))
    memory_task = asyncio.create_task(store.search_memory(user_query, user_id, top_k=3))
    facts_task = asyncio.create_task(store.search_facts(user_query, user_id, top_k=3))

    profile, memories, relevant_facts = await asyncio.gather(
        profile_task, memory_task, facts_task,
        return_exceptions=True,
    )

    parts: list[str] = []

    # Профиль пользователя
    if isinstance(profile, list) and profile:
        profile_lines = [f"  • [{f['fact_type']}] {f['text']}" for f in profile]
        parts.append("**Профиль пользователя:**\n" + "\n".join(profile_lines))

    # Релевантные факты
    if isinstance(relevant_facts, list) and relevant_facts:
        # Дедупликация с профилем
        profile_texts = {f["text"] for f in (profile if isinstance(profile, list) else [])}
        unique_facts = [f for f in relevant_facts if f["text"] not in profile_texts]
        if unique_facts:
            fact_lines = [f"  • {f['text']}" for f in unique_facts]
            parts.append("**Релевантные факты:**\n" + "\n".join(fact_lines))

    # Контекст диалогов
    if isinstance(memories, list) and memories:
        mem_lines = [
            f"  • {m['text'][:120]} (релевантность: {m['score']:.2f})"
            for m in memories
        ]
        parts.append("**Контекст из предыдущих диалогов:**\n" + "\n".join(mem_lines))

    return "\n\n".join(parts) if parts else ""
