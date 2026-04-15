"""
Точка входа FastAPI-приложения.

Lifespan управляет жизненным циклом:
  - Startup: запуск MCP-сервера, инициализация Search-графа
  - Shutdown: остановка MCP-сервера
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import os

from app.core.config import get_settings
from app.api.routes import router
from app.api.audio_routes import router as audio_router
from app.api.pptx_routes import router as pptx_router
from app.api.docx_routes import router as docx_router
from app.api.website_routes import router as website_router

settings = get_settings()

# Настройка логирования — детальный вывод
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

# Глушим только совсем болтливые библиотеки (НЕ uvicorn.access — он нам нужен)
for logger_name in ("httpx", "httpcore", "multipart", "httpcore.http11"):
    logging.getLogger(logger_name).setLevel(logging.WARNING)


logger = logging.getLogger("mts.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом: MCP + графы."""
    from app.services.web_tools import start_mcp, stop_mcp
    from app.graph.search_chat import init_search_graph

    print(f"🚀 {settings.APP_TITLE} v{settings.APP_VERSION} запущен")

    # 1. Запускаем MCP-сервер (subprocess stdio)
    await start_mcp()

    # 2. Инициализируем Search-граф (требует MCP tools)
    init_search_graph()

    print("✅ Все модули инициализированы")

    yield

    # Shutdown
    await stop_mcp()
    print("👋 Сервер завершает работу...")


app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Разрешаем CORS со всех источников (в т.ч. для OpenWebUI если он шлет запросы из браузера)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роутеры с префиксом /v1
app.include_router(router, prefix="/v1")
app.include_router(audio_router, prefix="/v1")   # ASR + Voice Chat
app.include_router(pptx_router, prefix="/v1")    # PPTX Generator
app.include_router(docx_router, prefix="/v1")    # DOCX Generator
app.include_router(website_router, prefix="/v1") # Website Generator

# Монтируем директорию со статикой, чтобы отдавать графики
os.makedirs("static/downloads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Редирект на документацию Swagger UI."""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    """Проверка работоспособности сервера."""
    from app.services.web_tools import is_available as mcp_available
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "mcp_available": mcp_available(),
    }
