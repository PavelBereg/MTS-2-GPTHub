
import asyncio
import logging
import uuid
import json
from datetime import datetime, timezone
from typing import Optional, NamedTuple

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, ScoredPoint
from langchain_openai import OpenAIEmbeddings

from app.core.config import get_settings

logger = logging.getLogger("mts.memory")

FACTS_COLLECTION = "user_facts_4k"

class ExtractedFact(NamedTuple):
    text: str
    fact_type: str

class MemoryStore:
    """Очищенный и стабильный Long-term Memory Store."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self.embeddings = OpenAIEmbeddings(
            base_url=self._settings.MWS_BASE_URL,
            api_key=self._settings.MWS_API_KEY,
            model=self._settings.EMBEDDING_MODEL,
        )
        # Используем СИНХРОННЫЙ клиент для надежности, вызывать будем через to_thread
        self.qclient = QdrantClient(url=self._settings.QDRANT_URL, timeout=10.0)
        self._collections_ready = False

    def _ensure_collections_sync(self):
        """Синхронная проверка коллекций."""
        if self._collections_ready:
            return
        try:
            collections_resp = self.qclient.get_collections()
            existing = {c.name for c in collections_resp.collections}
            
            if f"{self._settings.QDRANT_COLLECTION}_4k" not in existing:
                self.qclient.create_collection(
                    collection_name=f"{self._settings.QDRANT_COLLECTION}_4k",
                    vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
                )
            if FACTS_COLLECTION not in existing:
                self.qclient.create_collection(
                    collection_name=FACTS_COLLECTION,
                    vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
                )
            self._collections_ready = True
        except Exception as e:
            logger.error(f"Failed to ensure collections: {e}")

    async def _ensure_collections(self) -> None:
        await asyncio.to_thread(self._ensure_collections_sync)

    def _get_collection_name(self, namespace: tuple[str, ...]) -> str:
        if not namespace: return f"{self._settings.QDRANT_COLLECTION}_4k"
        return FACTS_COLLECTION if namespace[0] == "facts" else f"{self._settings.QDRANT_COLLECTION}_4k"

    # ── БАЗОВЫЕ ОПЕРАЦИИ (БЕЗОПАСНЫЕ ДЛЯ АСИНХРОННОСТИ) ──

    async def put(self, namespace: tuple[str, ...], key: str, value: dict) -> None:
        await self._ensure_collections()
        collection = self._get_collection_name(namespace)
        try:
            text = value.get("text", "")
            if not text: return

            vector = await self.embeddings.aembed_query(text)
            
            payload = {**value}
            if len(namespace) > 1: payload["user_id"] = namespace[1]
            if "timestamp" not in payload:
                payload["timestamp"] = datetime.now(timezone.utc).isoformat()

            await asyncio.to_thread(
                self.qclient.upsert,
                collection_name=collection,
                points=[PointStruct(id=key, vector=vector, payload=payload)],
            )
        except Exception as e:
            logger.error(f"Store put failed for {namespace}/{key}: {e}")

    async def search(self, namespace: tuple[str, ...], query: str, limit: int = 5, score_threshold: float = 0.65) -> list[dict]:
        if not query or not query.strip():
            return []
            
        await self._ensure_collections()
        collection = self._get_collection_name(namespace)
        try:
            query_vector = await self.embeddings.aembed_query(query)
            query_filter = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=namespace[1]))]) if len(namespace) > 1 else None

            # ИСПРАВЛЕНИЕ: .search() удалён в qdrant-client >= 1.7, используем .query_points()
            response = await asyncio.to_thread(
                self.qclient.query_points,
                collection_name=collection,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )
            return [{**hit.payload, "score": hit.score} for hit in response.points]
        except Exception as e:
            logger.error(f"Store search failed for {namespace}: {e}")
            return []

    # ── АЛИАСЫ ──
    async def add_memory(self, text: str, user_id: str) -> str:
        point_id = str(uuid.uuid4())
        await self.put(("memories", user_id), point_id, {"text": text, "type": "memory"})
        return point_id

    async def search_memory(self, query: str, user_id: str, top_k: int = 5) -> list[dict]:
        return await self.search(("memories", user_id), query, limit=top_k)

    async def search_facts(self, query: str, user_id: str, top_k: int = 5) -> list[dict]:
        return await self.search(("facts", user_id), query, limit=top_k)

    async def get_user_profile(self, user_id: str, top_k: int = 10) -> list[dict]:
        await self._ensure_collections()
        try:
            # ИСПРАВЛЕНИЕ: Вызов синхронного scroll в треде
            results = await asyncio.to_thread(
                self.qclient.scroll,
                collection_name=FACTS_COLLECTION,
                scroll_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
                limit=top_k,
                with_payload=True,
            )
            points, _ = results
            sorted_points = sorted(points, key=lambda p: p.payload.get("timestamp", ""), reverse=True)
            return [{"text": p.payload.get("text", ""), "fact_type": p.payload.get("fact_type", "general")} for p in sorted_points]
        except Exception:
            return []

# Инициализируем глобально, без всяких LazyProxy
memory_store = MemoryStore()

# ═══════════════════════════════════════════════════════════════════
# ИЗВЛЕЧЕНИЕ ФАКТОВ (Перенесено сюда, чтобы убрать circular imports)
# ═══════════════════════════════════════════════════════════════════

async def extract_and_save_facts(user_message: str, user_id: str) -> None:
    """Фоновая задача извлечения фактов."""
    if not user_message or len(user_message.strip()) < 10: return
    
    settings = get_settings()
    prompt = f"""Извлеки факты ДЛЯ ЗАПОМИНАНИЯ о пользователе. Верни ТОЛЬКО валидный JSON массив: [{{"fact": "...", "type": "name|project|preference|context"}}]
    Сообщение: "{user_message[:800]}\""""
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{settings.MWS_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {settings.MWS_API_KEY}"},
                json={"model": settings.LLM_MODEL_ALT, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 150}
            )
            if resp.status_code != 200:
                logger.debug(f"Fact extraction skipped: API status {resp.status_code}")
                return
                
            data = resp.json()
            if "choices" not in data or not data["choices"]:
                return
            raw = data["choices"][0]["message"]["content"]
            
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end == -1: return
        
        facts = json.loads(raw[start : end + 1])
        for item in facts:
            if isinstance(item, dict) and item.get("fact"):
                await memory_store.put(("facts", user_id), str(uuid.uuid4()), {"text": item["fact"], "fact_type": item.get("type", "general")})
    except Exception as e:
        logger.debug(f"Fact extraction skipped/failed: {e}")

async def build_memory_context(user_query: str, user_id: str) -> str:
    """Сборка контекста из памяти."""
    profile_task = asyncio.create_task(memory_store.get_user_profile(user_id, top_k=5))
    memory_task = asyncio.create_task(memory_store.search_memory(user_query, user_id, top_k=3))
    
    profile, memories = await asyncio.gather(profile_task, memory_task, return_exceptions=True)
    parts = []
    
    if isinstance(profile, list) and profile:
        parts.append("ФАКТЫ О ПОЛЬЗОВАТЕЛЕ:\n" + "\n".join([f"- {f['text']}" for f in profile]))
    if isinstance(memories, list) and memories:
        parts.append("ПРОШЛЫЕ ДИАЛОГИ:\n" + "\n".join([f"- {m['text'][:150]}" for m in memories]))
        
    return "\n\n".join(parts)