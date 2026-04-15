"""
Конфигурация приложения.

Все настройки загружаются из переменных окружения (или .env файла).
Имена моделей MWS API вынесены как константы.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Центральный конфиг бэкенда."""

    # ── MWS API ──────────────────────────────────────────────────────────
    MWS_API_KEY: str = "CHANGE_ME"
    MWS_BASE_URL: str = "https://api.gpt.mws.ru/v1"

    # ── Имена моделей MWS ────────────────────────────────────────────────
    LLM_MODEL: str = "llama-3.3-70b-instruct"          # Главный мозг / агент (Топовая!)
    LLM_MODEL_ALT: str = "llama-3.1-8b-instruct"       # Быстрый роутер / классификатор
    LLM_MODEL_LIGHT: str = "qwen2.5-72b-instruct"      # Качественный чат
    IMAGE_MODEL: str = "qwen-image-lightning"          # Быстрая генерация картинок
    ASR_MODEL: str = "whisper-turbo-local"             # Распознавание речи
    VLM_MODEL: str = "qwen2.5-vl-72b"                  # Лучшее зрение
    EMBEDDING_MODEL: str = "qwen3-embedding-8b"        # Эмбеддинги для памяти
    CODER_MODEL: str = "qwen3-coder-480b-a35b"         # Спец. модель для кода и графиков

    # ── Research / Search ─────────────────────────────────────────────
    RESEARCH_MAX_QUESTIONS: int = 5                    # Макс. подвопросов в Deep Research
    RESEARCH_MAX_URLS: int = 3                         # Макс. URL-ов на подвопрос
    SEARCH_MAX_RESULTS: int = 5                        # Макс. результатов обычного поиска

    # ── Qdrant ───────────────────────────────────────────────────────────
    QDRANT_URL: str = "http://qdrant:6333"                 # URL сервиса Qdrant в docker-compose
    QDRANT_COLLECTION: str = "user_memory"             # Коллекция по умолчанию

    # ── Приложение ───────────────────────────────────────────────────────
    APP_TITLE: str = "MTS AI Assistant Backend"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    BACKEND_URL: str = "http://localhost:8000"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# ── Константы режимов ("виртуальные модели" в OpenWebUI) ──────────────
MODE_CHAT = "💬 Обычный чат"
MODE_AUTO = "🌟 AUTO (Умный роутер МТС)"
MODE_SEARCH = "🔍 Поиск + Чат"
MODE_RESEARCH = "🔬 Deep Research"
MODE_PRESENTATION = "📊 Презентации"
MODE_DOCUMENT = "📄 Документы Word"


@lru_cache
def get_settings() -> Settings:
    """Singleton-фабрика для настроек (кэшируется)."""
    return Settings()
