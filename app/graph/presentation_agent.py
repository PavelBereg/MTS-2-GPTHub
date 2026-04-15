"""
LangGraph агент для интерактивной подготовки PPTX.

Задача графа — не генерировать файл сразу, а провести пользователя через
короткий брифинг:
  1. собрать тему, аудиторию, цель, стиль и количество слайдов;
  2. показать структуру слайдов;
  3. дождаться явного подтверждения;
  4. передать согласованную структуру в python-pptx генератор.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Annotated, Any, AsyncGenerator, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger("mts.presentation_agent")


PresentationStage = Literal[
    "collecting_brief",
    "brief_review",
    "outline_review",
    "ready_to_generate",
    "generated",
]


class PresentationBrief(TypedDict, total=False):
    """Бриф презентации, который постепенно собирается из диалога."""

    topic: str
    audience: str
    goal: str
    style: str
    slides_count: int
    language: str
    duration_minutes: int
    key_messages: list[str]
    must_include: list[str]
    must_avoid: list[str]
    source_material: str
    deck_title: str
    deck_subtitle: str


class PresentationSlide(TypedDict, total=False):
    """Один слайд согласованной структуры."""

    title: str
    purpose: str
    bullets: list[str]
    visual_hint: str
    speaker_notes: str
    layout: str
    highlight: str
    body: str
    evidence: list[str]
    takeaway: str
    icon: str


class PresentationState(TypedDict):
    """Состояние интерактивного агента презентаций."""

    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    session_id: str
    model: str
    memory_context: str
    last_user_text: str

    brief: PresentationBrief
    missing_fields: list[str]
    last_question_field: str
    stage: PresentationStage
    brief_confirmed: bool

    outline: list[PresentationSlide]
    outline_confirmed: bool
    file_url: str
    status: str


REQUIRED_FIELDS: tuple[str, ...] = (
    "topic",
    "audience",
    "goal",
    "style",
    "slides_count",
)

FIELD_LABELS: dict[str, str] = {
    "topic": "тема",
    "audience": "целевая аудитория",
    "goal": "цель презентации",
    "style": "стиль",
    "slides_count": "количество слайдов",
}

_PRESENTATION_SESSIONS: dict[str, dict[str, Any]] = {}
_SESSION_TTL_SECONDS = 60 * 60 * 3

_CONFIRM_RE = re.compile(
    r"^\s*(да|ок|окей|подтверждаю|утверждаю|всё верно|все верно|"
    r"поехали|давай|начинай|старт|го|go|делай|создавай|генерируй|можно создавать|"
    r"структура подходит|подходит|согласен)\s*[.!?]*\s*$",
    re.IGNORECASE,
)

_CANCEL_RE = re.compile(
    r"^\s*(отмена|отмени|стоп|stop|cancel|не надо|закрой режим)\s*[.!?]*\s*$",
    re.IGNORECASE,
)

_NEGATIVE_RE = re.compile(
    r"\b(измени|поменяй|добавь|убери|замени|не подходит|не то|лучше|"
    r"переделай|пересобери|исправь)\b",
    re.IGNORECASE,
)


BRIEF_EXTRACT_PROMPT = """\
Ты — эксперт по подготовке деловых презентаций.

Извлеки из диалога параметры будущей презентации. Не придумывай значения,
если пользователь их не сообщил явно.

Текущий бриф, если он уже есть:
{brief_json}

Диалог:
{transcript}

Верни ТОЛЬКО валидный JSON без markdown:
{{
  "topic": string|null,
  "audience": string|null,
  "goal": string|null,
  "style": string|null,
  "slides_count": integer|null,
  "language": "ru"|"en"|null,
  "duration_minutes": integer|null,
  "key_messages": string[],
  "must_include": string[],
  "must_avoid": string[],
  "source_material": string|null
}}

Правила:
- style нормализуй в одно из значений:
  "mts_corporate", "strict_corporate", "modern_light", "editorial", "tech", "creative", "dark", "minimal".
- Если пользователь явно просит стиль МТС или корпоративный МТС, выбирай "mts_corporate".
- Если пользователь говорит "строго", "для руководства", "совет директоров",
  выбирай "strict_corporate", если нет других указаний.
- Если пользователь говорит "креативно", "ярко", "необычно", выбирай "creative".
- Если пользователь прислал большой фрагмент текста/заметок/сырой материал для презентации,
  сохрани его смысл в source_material. Не выбрасывай конкретику.
- Если число слайдов не указано, slides_count = null.
- Если язык не указан, language = "ru".
- Никакого текста вне JSON.
"""

OUTLINE_PROMPT = """\
Ты — senior presentation strategist. Составь сильную структуру презентации.

Бриф:
{brief_json}

Контекст из памяти пользователя:
{memory_context}

{revision_block}

Верни ТОЛЬКО валидный JSON без markdown:
{{
  "presentation_title": "...",
  "subtitle": "...",
  "slides": [
    {{
      "title": "...",
      "purpose": "зачем этот слайд нужен",
      "layout": "hero|split|cards|timeline|comparison|process|quote|takeaway",
      "highlight": "главная крупная фраза слайда, 6-14 слов",
      "body": "2-3 предложения с обработанным содержанием слайда",
      "evidence": ["конкретный факт из материала", "ещё один факт"],
      "bullets": ["...", "...", "..."],
      "visual_hint": "конкретная идея визуала, схемы, стрелок, карточек или иконок",
      "icon": "одно короткое слово для иконки: target|people|data|risk|idea|growth|shield|map|clock|check",
      "takeaway": "один вывод слайда, 8-16 слов",
      "speaker_notes": "что сказать голосом"
    }}
  ],
  "closing_message": "..."
}}

Правила:
- presentation_title: короткое редакторское название, не более 9 слов. Не копируй сырой запрос пользователя.
- subtitle: короткое пояснение, не более 12 слов. Не вставляй туда длинную цель или аудиторию.
- Количество слайдов строго равно brief.slides_count.
- Каждый слайд должен вести аудиторию к цели презентации.
- Заголовки не должны быть вопросами. Формулируй их как смысловые утверждения.
- Не используй пустые шаблонные заголовки вроде "Введение" или "Обзор".
- Учитывай аудиторию, цель, стиль и обязательные тезисы.
- Если brief.source_material есть, строи содержание прежде всего по нему.
- Запрещены generic-буллиты вроде "имеет богатую историю", "наследие важно", "показать связи".
- Каждый буллит должен быть содержательным: действие, аргумент, наблюдение, причинно-следственная связь или вывод.
- body должен отвечать на вопрос "что именно мы хотим донести на этом слайде", а не повторять заголовок.
- evidence заполняй конкретными фактами из source_material. Если source_material нет, укажи аргументы/наблюдения, но не выдумывай числа.
- Делай разные типы слайдов: не более двух подряд с одинаковым layout.
- Буллиты короткие: 2-4 на слайд, до 13 слов каждый.
- highlight — не повтор заголовка, а сильная фраза для крупной вставки.
- takeaway — финальный смысл слайда, который можно вынести в плашку.
- visual_hint должен помогать дизайнеру понять визуальную композицию.
- Для технической аудитории давай больше схем, процессов, сравнений и причинно-следственных связей.
"""

ENRICH_OUTLINE_PROMPT = """\
Ты — редактор содержательных презентаций. Твоя задача — усилить уже составленную
структуру так, чтобы каждый слайд нес смысл, опирался на материал пользователя
и не выглядел набором вопросов или однословных тезисов.

Бриф:
{brief_json}

Текущая структура:
{outline_json}

Верни ТОЛЬКО валидный JSON без markdown:
{{
  "slides": [
    {{
      "title": "...",
      "purpose": "...",
      "layout": "hero|split|cards|timeline|comparison|process|quote|takeaway",
      "highlight": "сильная мысль, не вопрос",
      "body": "2-3 предложения с обработанным смыслом слайда",
      "evidence": ["факт или деталь из source_material", "второй факт или деталь"],
      "bullets": ["содержательный тезис", "аргумент", "вывод"],
      "visual_hint": "...",
      "icon": "target|people|data|risk|idea|growth|shield|map|clock|check",
      "takeaway": "вывод слайда",
      "speaker_notes": "что сказать голосом"
    }}
  ],
  "presentation_title": "короткое название презентации, не более 9 слов",
  "subtitle": "короткое пояснение, не более 12 слов",
  "closing_message": "3-4 предложения: итоговая мысль всей презентации, не цель и не 'спасибо'"
}}

Жёсткие правила:
- presentation_title и subtitle должны быть отредактированы, а не скопированы из сырого брифа.
- Все длинные пользовательские формулировки надо сжать до ясных тезисов.
- Если есть source_material, минимум 70% смысла должно идти из него.
- Не оставляй на слайде только вопросы. Любой вопрос преобразуй в утверждение или объяснение.
- Заголовки не должны заканчиваться вопросительным знаком.
- Не используй пустые формулы: "важно", "уникальные способности", "сделать выводы", если не раскрываешь почему.
- Каждый слайд должен иметь: highlight, body, 2-4 bullets, takeaway.
- body должен быть конкретнее bullets и раскрывать контекст.
- body: максимум 190 символов.
- evidence: каждый пункт максимум 95 символов.
- bullets: каждый пункт максимум 90 символов.
- highlight: максимум 85 символов.
- takeaway: максимум 100 символов.
- evidence — только факты/детали из материала пользователя или явно помеченные выводы из него.
- closing_message должен суммировать факты и аргументы презентации. Запрещено возвращать одно слово.
"""


def _now() -> float:
    return time.time()


def _purge_expired_sessions() -> None:
    cutoff = _now() - _SESSION_TTL_SECONDS
    expired = [
        key
        for key, value in _PRESENTATION_SESSIONS.items()
        if value.get("updated_at", 0) < cutoff
    ]
    for key in expired:
        _PRESENTATION_SESSIONS.pop(key, None)


def has_active_presentation_session(session_id: str) -> bool:
    """Есть ли незавершённый PPTX-брифинг для этого чата."""
    _purge_expired_sessions()
    session = _PRESENTATION_SESSIONS.get(session_id)
    if not session:
        return False
    return session.get("stage") != "generated"


def clear_presentation_session(session_id: str) -> None:
    """Сбрасывает активный PPTX-брифинг."""
    _PRESENTATION_SESSIONS.pop(session_id, None)


def _get_session(session_id: str) -> dict[str, Any]:
    _purge_expired_sessions()
    session = _PRESENTATION_SESSIONS.get(session_id)
    if not session:
        session = {
            "brief": {},
            "outline": [],
            "stage": "collecting_brief",
            "last_question_field": "",
            "updated_at": _now(),
        }
        _PRESENTATION_SESSIONS[session_id] = session
    return session


def _save_session(session_id: str, state: PresentationState) -> None:
    _PRESENTATION_SESSIONS[session_id] = {
        "brief": dict(state.get("brief", {})),
        "outline": list(state.get("outline", [])),
        "stage": state.get("stage", "collecting_brief"),
        "last_question_field": state.get("last_question_field", ""),
        "updated_at": _now(),
    }


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif "text" in block:
                    parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def _last_human_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return _stringify_content(msg.content).strip()
    return ""


def _messages_to_transcript(messages: list[BaseMessage], limit: int = 16) -> str:
    rows: list[str] = []
    for msg in messages[-limit:]:
        if isinstance(msg, HumanMessage):
            role = "Пользователь"
        elif isinstance(msg, AIMessage):
            role = "Ассистент"
        else:
            role = "Система"
        text = _stringify_content(msg.content).strip()
        if text:
            rows.append(f"{role}: {text}")
    return "\n".join(rows)


def _strip_think_and_fences(text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text or "").strip()
    text = re.sub(r"```(?:json)?\s*", "", text).strip("` \n")
    return text


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = _strip_think_and_fences(text)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        logger.warning("Presentation JSON parse failed: %r", raw[:300])
        return {}
    return obj if isinstance(obj, dict) else {}


def _normalize_style(value: Any) -> str | None:
    if not value:
        return None
    t = str(value).lower().strip()
    if any(k in t for k in ("мтс", "mts", "бренд", "brand")):
        return "mts_corporate"
    if any(k in t for k in ("strict", "строг", "руковод", "директор", "топ", "board")):
        return "strict_corporate"
    if any(k in t for k in ("tech", "тех", "схем", "архитект", "инженер")):
        return "tech"
    if any(k in t for k in ("editorial", "журнал", "редакц", "сторител", "story")):
        return "editorial"
    if any(k in t for k in ("modern", "современ", "нейтральн", "чист")):
        return "modern_light"
    if any(k in t for k in ("creative", "креатив", "ярк", "необыч", "смел")):
        return "creative"
    if any(k in t for k in ("minimal", "минимал")):
        return "minimal"
    if any(k in t for k in ("dark", "темн", "тёмн")):
        return "dark"
    if any(k in t for k in ("corporate", "корпоратив", "делов")):
        return "strict_corporate"
    return str(value).strip()


def _normalize_brief(raw: dict[str, Any]) -> PresentationBrief:
    brief: PresentationBrief = {}

    for key in ("topic", "audience", "goal", "language", "source_material"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            brief[key] = value.strip()

    style = _normalize_style(raw.get("style"))
    if style:
        brief["style"] = style

    slides_count = raw.get("slides_count")
    if isinstance(slides_count, str):
        match = re.search(r"\d+", slides_count)
        slides_count = int(match.group()) if match else None
    if isinstance(slides_count, (int, float)):
        brief["slides_count"] = max(2, min(int(slides_count), 15))

    duration = raw.get("duration_minutes")
    if isinstance(duration, str):
        match = re.search(r"\d+", duration)
        duration = int(match.group()) if match else None
    if isinstance(duration, (int, float)):
        brief["duration_minutes"] = max(1, min(int(duration), 180))

    for key in ("key_messages", "must_include", "must_avoid"):
        value = raw.get(key)
        if isinstance(value, list):
            items = [str(v).strip() for v in value if str(v).strip()]
            if items:
                brief[key] = items[:10]

    if "language" not in brief:
        brief["language"] = "ru"

    return brief


def _merge_brief(old: PresentationBrief, new: PresentationBrief) -> PresentationBrief:
    merged: PresentationBrief = dict(old)
    for key, value in new.items():
        if value not in (None, "", [], {}):
            merged[key] = value
    return merged


def _missing_fields(brief: PresentationBrief) -> list[str]:
    missing: list[str] = []
    for field in REQUIRED_FIELDS:
        value = brief.get(field)
        if value in (None, "", [], {}):
            missing.append(field)
    return missing


def _extract_topic_from_request(text: str) -> str:
    t = text.strip()
    t = re.sub(
        r"^\s*(сделай|создай|подготовь|сгенерируй|собери)\s+",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"\b(презентацию|презентация|слайды|pptx|powerpoint)\b",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"\b(на тему|по теме|про|о)\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip(" .,:;-")
    return t


def _fallback_extract(text: str, current_brief: PresentationBrief) -> PresentationBrief:
    """Детерминированный минимум, если LLM недоступна или дала плохой JSON."""
    extracted: PresentationBrief = {}
    t = text.lower()

    if "topic" not in current_brief and re.search(r"презентац|слайд|pptx|powerpoint", t):
        topic = _extract_topic_from_request(text)
        if len(topic) >= 3:
            extracted["topic"] = topic

    slides_match = re.search(r"(\d{1,2})\s*(?:слайд|slide)", t)
    if slides_match:
        extracted["slides_count"] = max(2, min(int(slides_match.group(1)), 15))

    style = _normalize_style(text)
    if style in {
        "mts_corporate",
        "strict_corporate",
        "modern_light",
        "editorial",
        "tech",
        "minimal",
        "creative",
        "dark",
    }:
        extracted["style"] = style

    audience_match = re.search(r"\bдля\s+([^.,;!?]+)", text, re.IGNORECASE)
    if audience_match:
        audience = audience_match.group(1).strip()
        if 3 <= len(audience) <= 120:
            extracted["audience"] = audience

    goal_match = re.search(r"\bцель[:\s-]+([^.;!?]+)", text, re.IGNORECASE)
    if goal_match:
        goal = goal_match.group(1).strip()
        if 3 <= len(goal) <= 180:
            extracted["goal"] = goal

    if len(text) > 120 and not re.search(r"^\s*(сделай|создай|подготовь|сгенерируй)", t):
        extracted["source_material"] = text[:5000].strip()

    extracted["language"] = "ru"
    return extracted


def _coerce_answer_for_field(field: str, text: str) -> PresentationBrief:
    """Если мы только что спросили конкретное поле, короткий ответ мапим в него."""
    clean = text.strip(" \n\t.,;")
    if not clean or _CANCEL_RE.match(clean):
        return {}
    if len(clean) > 300:
        clean = clean[:300].strip()

    if field == "slides_count":
        match = re.search(r"\d{1,2}", clean)
        if match:
            return {"slides_count": max(2, min(int(match.group()), 15))}
        return {}

    if field == "style":
        style = _normalize_style(clean)
        return {"style": style} if style else {}

    if field in {"topic", "audience", "goal"} and len(clean) >= 2:
        return {field: clean}

    return {}


async def _extract_brief_with_llm(
    messages: list[BaseMessage],
    current_brief: PresentationBrief,
    model: str,
) -> PresentationBrief:
    transcript = _messages_to_transcript(messages)
    prompt = BRIEF_EXTRACT_PROMPT.format(
        brief_json=json.dumps(current_brief, ensure_ascii=False),
        transcript=transcript,
    )
    llm = ChatOpenAI(
        base_url=settings.MWS_BASE_URL,
        api_key=settings.MWS_API_KEY,
        model=model or settings.LLM_MODEL,
        temperature=0.1,
        streaming=False,
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return _normalize_brief(_extract_json_object(str(response.content)))


def _is_outline_confirmation(text: str) -> bool:
    clean = text.strip()
    if not clean:
        return False
    if _NEGATIVE_RE.search(clean):
        return False
    return bool(_CONFIRM_RE.match(clean))


def _make_question(field: str, brief: PresentationBrief) -> str:
    topic = brief.get("topic", "презентации")
    templates = {
        "topic": (
            "Давай сделаем это как нормальный презентационный бриф. "
            "Какая точная тема презентации? Если у тебя уже есть материалы, текст, "
            "факты или заметки, их можно загрузить или прислать сюда — я соберу презентацию по ним."
        ),
        "audience": (
            f"Понял тему: **{topic}**. Для кого готовим презентацию: "
            "топ-менеджмент, техническая команда, продажи, клиенты или другая аудитория?"
        ),
        "goal": (
            "Какую цель должна решить презентация: убедить, обучить, продать идею, "
            "защитить проект, показать статус или что-то другое?"
        ),
        "style": (
            "Какой стиль выбрать? Например: **МТС-корпоративный**, строгий деловой, "
            "**tech-схемы**, редакционный сторителлинг, минимализм, тёмный или креативный."
        ),
        "slides_count": (
            "Сколько примерно слайдов нужно? Можно ответить числом, например: **7**. "
            "После этого можно докинуть материалы или текст, чтобы слайды опирались на факты."
        ),
    }
    return templates.get(field, f"Уточни, пожалуйста: {FIELD_LABELS.get(field, field)}?")


def _normalize_outline(raw: dict[str, Any], brief: PresentationBrief) -> list[PresentationSlide]:
    slides_raw = raw.get("slides", [])
    if not isinstance(slides_raw, list):
        return []

    slides: list[PresentationSlide] = []
    for idx, item in enumerate(slides_raw, 1):
        if not isinstance(item, dict):
            continue
        bullets_raw = item.get("bullets", [])
        bullets = [str(b).strip() for b in bullets_raw if str(b).strip()] if isinstance(bullets_raw, list) else []
        slides.append(
            {
                "title": str(item.get("title") or f"Слайд {idx}").strip(),
                "purpose": str(item.get("purpose") or "").strip(),
                "bullets": bullets[:5],
                "visual_hint": str(item.get("visual_hint") or "").strip(),
                "speaker_notes": str(item.get("speaker_notes") or item.get("notes") or "").strip(),
                "layout": str(item.get("layout") or "").strip(),
                "highlight": str(item.get("highlight") or "").strip(),
                "body": str(item.get("body") or "").strip(),
                "evidence": [
                    str(e).strip()
                    for e in item.get("evidence", [])
                    if str(e).strip()
                ][:4]
                if isinstance(item.get("evidence", []), list)
                else [],
                "takeaway": str(item.get("takeaway") or "").strip(),
                "icon": str(item.get("icon") or "").strip(),
            }
        )

    target_count = int(brief.get("slides_count", len(slides) or 5))
    return slides[:target_count]


def _shorten(text: Any, limit: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    cut = value[: max(0, limit - 1)].rstrip()
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(".,;:") + "…"


def _compact_slide_text(slide: PresentationSlide) -> PresentationSlide:
    item = dict(slide)
    item["title"] = _shorten(item.get("title"), 58)
    item["highlight"] = _shorten(item.get("highlight"), 85)
    item["body"] = _shorten(item.get("body"), 190)
    item["takeaway"] = _shorten(item.get("takeaway"), 100)
    item["purpose"] = _shorten(item.get("purpose"), 120)
    item["visual_hint"] = _shorten(item.get("visual_hint"), 130)
    item["bullets"] = [_shorten(b, 90) for b in item.get("bullets", []) if str(b).strip()][:4]
    item["evidence"] = [_shorten(e, 95) for e in item.get("evidence", []) if str(e).strip()][:3]
    return item


def _render_outline(outline: list[PresentationSlide], brief: PresentationBrief) -> str:
    lines = [
        "Я собрал структуру презентации. Проверь, пожалуйста:",
        "",
        f"**Тема:** {brief.get('topic', '—')}",
        f"**Аудитория:** {brief.get('audience', '—')}",
        f"**Цель:** {brief.get('goal', '—')}",
        f"**Стиль:** {brief.get('style', '—')}",
        "",
    ]
    for idx, slide in enumerate(outline, 1):
        lines.append(f"**{idx}. {slide.get('title', f'Слайд {idx}')}**")
        highlight = slide.get("highlight")
        if highlight:
            lines.append(f"Главная мысль: {highlight}")
        body = slide.get("body")
        if body:
            lines.append(f"Содержание: {body[:260]}")
        purpose = slide.get("purpose")
        if purpose:
            lines.append(f"Задача: {purpose}")
        bullets = slide.get("bullets") or []
        for bullet in bullets[:3]:
            lines.append(f"- {bullet}")
        evidence = slide.get("evidence") or []
        if evidence:
            lines.append("Факты: " + "; ".join(evidence[:2]))
        visual = slide.get("visual_hint")
        if visual:
            lines.append(f"Визуал: {visual}")
        lines.append("")

    lines.append(
        "Если всё подходит, напиши **«подтверждаю»** — после этого я создам PPTX. "
        "Если нужно поправить структуру, напиши, что изменить."
    )
    return "\n".join(lines).strip()


async def _enrich_outline_with_llm(
    brief: PresentationBrief,
    outline: list[PresentationSlide],
    model: str,
) -> tuple[list[PresentationSlide], str]:
    prompt = ENRICH_OUTLINE_PROMPT.format(
        brief_json=json.dumps(brief, ensure_ascii=False),
        outline_json=json.dumps(outline, ensure_ascii=False),
    )
    llm = ChatOpenAI(
        base_url=settings.MWS_BASE_URL,
        api_key=settings.MWS_API_KEY,
        model=model or settings.LLM_MODEL,
        temperature=0.25,
        streaming=False,
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    obj = _extract_json_object(str(response.content))
    enriched = _normalize_outline(obj, brief)
    closing = str(obj.get("closing_message") or "").strip()
    if obj.get("presentation_title"):
        brief["deck_title"] = _shorten(obj.get("presentation_title"), 78)
    if obj.get("subtitle"):
        brief["deck_subtitle"] = _shorten(obj.get("subtitle"), 90)
    return enriched or outline, closing


def _repair_outline_content(
    outline: list[PresentationSlide],
    brief: PresentationBrief,
) -> list[PresentationSlide]:
    """Детерминированно заполняет пустые смысловые поля, если LLM сдала слабый JSON."""
    source = str(brief.get("source_material") or "").strip()
    source_sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n+", source)
        if len(s.strip()) > 24
    ]
    topic = str(brief.get("topic") or "тема")
    goal = str(brief.get("goal") or "донести ключевую идею")

    repaired: list[PresentationSlide] = []
    for idx, slide in enumerate(outline):
        item = dict(slide)
        title = str(item.get("title") or f"Слайд {idx + 1}").strip()
        if title.endswith("?"):
            title = title.rstrip("?").strip()
            if title.lower().startswith(("почему ", "зачем ", "как ", "что ")):
                title = f"Смысл: {title}"
            item["title"] = title
        bullets = [str(b).strip() for b in item.get("bullets", []) if str(b).strip()]

        if len(bullets) < 2:
            fact = source_sentences[idx % len(source_sentences)] if source_sentences else ""
            bullets = [
                fact[:130] if fact else f"{topic}: раскрыть ключевой аспект {idx + 1}",
                f"Связать этот аспект с задачей: {goal}",
                "Показать, какой вывод должна сделать аудитория",
            ]

        if not item.get("highlight"):
            item["highlight"] = bullets[0][:120]
        if not item.get("body"):
            fact = source_sentences[idx % len(source_sentences)] if source_sentences else bullets[0]
            item["body"] = (
                f"{fact} Это важно для презентации, потому что помогает аудитории "
                f"понять не только факт, но и его роль в общей логике темы."
            )[:420]
        if not item.get("evidence"):
            item["evidence"] = source_sentences[idx : idx + 2] or bullets[:2]
        if not item.get("takeaway"):
            item["takeaway"] = (
                f"Вывод: {bullets[-1]}"
                if not str(bullets[-1]).lower().startswith("вывод")
                else bullets[-1]
            )[:160]
        if not item.get("purpose"):
            item["purpose"] = f"Раскрыть смысл блока «{title}» через факты и вывод."
        if not item.get("layout"):
            item["layout"] = ["hero", "split", "cards", "process", "comparison", "quote", "takeaway"][idx % 7]
        if not item.get("icon"):
            item["icon"] = ["target", "data", "idea", "growth", "map", "check"][idx % 6]

        item["bullets"] = bullets[:4]
        repaired.append(_compact_slide_text(item))

    return repaired


def _render_brief_review(brief: PresentationBrief) -> str:
    style_labels = {
        "mts_corporate": "МТС-корпоративный",
        "strict_corporate": "строгий деловой",
        "modern_light": "современный светлый",
        "editorial": "редакционный сторителлинг",
        "tech": "tech-схемы",
        "creative": "креативный",
        "dark": "тёмный",
        "minimal": "минимализм",
    }
    style_label = style_labels.get(str(brief.get("style") or ""), brief.get("style", "—"))
    lines = [
        "Бриф собран. Проверь, всё ли верно:",
        "",
        f"**Тема:** {brief.get('topic', '—')}",
        f"**Аудитория:** {brief.get('audience', '—')}",
        f"**Цель:** {brief.get('goal', '—')}",
        f"**Стиль:** {style_label}",
        f"**Слайдов:** {brief.get('slides_count', '—')}",
    ]
    if brief.get("duration_minutes"):
        lines.append(f"**Длительность выступления:** {brief['duration_minutes']} мин.")
    if brief.get("key_messages"):
        lines.append("**Ключевые мысли:** " + "; ".join(brief["key_messages"]))
    if brief.get("must_include"):
        lines.append("**Обязательно включить:** " + "; ".join(brief["must_include"]))
    if brief.get("must_avoid"):
        lines.append("**Не использовать:** " + "; ".join(brief["must_avoid"]))
    if brief.get("source_material"):
        source = str(brief["source_material"])
        preview = source[:280] + ("..." if len(source) > 280 else "")
        lines.append(f"**Исходный материал:** {preview}")

    lines += [
        "",
        "Если бриф верный, напиши **«поехали»** — тогда я составлю структуру слайдов.",
        "Можно также загрузить/прислать текст, тезисы или факты — я встрою их в содержание слайдов.",
        "Если нужно поправить вводные, напиши правку обычным текстом.",
    ]
    return "\n".join(lines)


async def extract_brief_node(state: PresentationState) -> dict:
    """Обновляет бриф по текущей реплике пользователя и истории диалога."""
    last_user = state.get("last_user_text") or _last_human_text(state["messages"])
    brief = dict(state.get("brief", {}))
    stage = state.get("stage", "collecting_brief")

    if stage == "outline_review" and state.get("outline"):
        confirmed = _is_outline_confirmation(last_user)
        if confirmed:
            return {
                "outline_confirmed": True,
                "missing_fields": [],
                "stage": "ready_to_generate",
                "status": "Структура подтверждена, создаю PPTX...",
            }
        if last_user:
            source_update = {}
            if len(last_user) > 80:
                previous_source = str(brief.get("source_material") or "")
                source_update["source_material"] = (
                    previous_source + "\n\n" + last_user
                ).strip()[-9000:]
            brief = _merge_brief(brief, source_update)
            return {
                "brief": brief,
                "brief_confirmed": True,
                "outline_confirmed": False,
                "missing_fields": [],
                "last_question_field": "",
                "stage": "outline_review",
                "status": "Пересобираю структуру с учётом правок...",
            }

    if stage == "brief_review" and _is_outline_confirmation(last_user):
        return {
            "brief_confirmed": True,
            "outline_confirmed": False,
            "missing_fields": [],
            "last_question_field": "",
            "stage": "outline_review",
            "status": "Бриф подтверждён, собираю структуру слайдов...",
        }

    last_question_field = state.get("last_question_field", "")
    direct_update = (
        _coerce_answer_for_field(last_question_field, last_user)
        if last_question_field
        else {}
    )

    try:
        llm_update = await _extract_brief_with_llm(
            state["messages"],
            brief,
            state.get("model", settings.LLM_MODEL),
        )
    except Exception:
        logger.exception("Presentation brief extraction via LLM failed")
        llm_update = {}

    fallback_update = _fallback_extract(last_user, brief)
    brief = _merge_brief(brief, fallback_update)
    brief = _merge_brief(brief, llm_update)
    brief = _merge_brief(brief, direct_update)

    missing = _missing_fields(brief)
    next_stage: PresentationStage = "collecting_brief" if missing else "brief_review"

    return {
        "brief": brief,
        "missing_fields": missing,
        "brief_confirmed": False,
        "outline_confirmed": False,
        "stage": next_stage,
    }


async def ask_question_node(state: PresentationState) -> dict:
    """Задаёт один следующий вопрос, без генерации структуры."""
    missing = state.get("missing_fields", [])
    field = missing[0] if missing else ""
    question = _make_question(field, state.get("brief", {}))
    return {
        "last_question_field": field,
        "status": question,
        "messages": [AIMessage(content=question)],
    }


async def show_brief_node(state: PresentationState) -> dict:
    """Показывает собранный бриф и ждёт отмашку на создание структуры."""
    message = _render_brief_review(state.get("brief", {}))
    return {
        "stage": "brief_review",
        "last_question_field": "",
        "status": message,
        "messages": [AIMessage(content=message)],
    }


async def make_outline_node(state: PresentationState) -> dict:
    """Генерирует или пересобирает структуру и отдаёт её на подтверждение."""
    brief = state.get("brief", {})
    last_user = state.get("last_user_text", "")
    current_outline = state.get("outline", [])

    revision_block = ""
    if current_outline and last_user and not _is_outline_confirmation(last_user):
        revision_block = (
            "Текущая структура уже была предложена:\n"
            f"{json.dumps(current_outline, ensure_ascii=False)}\n\n"
            "Пользователь попросил изменить её так:\n"
            f"{last_user}\n\n"
            "Пересобери структуру с учётом правок."
        )

    prompt = OUTLINE_PROMPT.format(
        brief_json=json.dumps(brief, ensure_ascii=False),
        memory_context=state.get("memory_context", "") or "нет",
        revision_block=revision_block,
    )

    llm = ChatOpenAI(
        base_url=settings.MWS_BASE_URL,
        api_key=settings.MWS_API_KEY,
        model=state.get("model", settings.LLM_MODEL),
        temperature=0.35,
        streaming=False,
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        obj = _extract_json_object(str(response.content))
        outline = _normalize_outline(obj, brief)
        closing_message = str(obj.get("closing_message") or "").strip()
        brief = dict(brief)
        if obj.get("presentation_title"):
            brief["deck_title"] = _shorten(obj.get("presentation_title"), 78)
        if obj.get("subtitle"):
            brief["deck_subtitle"] = _shorten(obj.get("subtitle"), 90)
    except Exception:
        logger.exception("Presentation outline generation failed")
        outline = []
        closing_message = ""

    if not outline:
        slides_count = int(brief.get("slides_count", 5))
        topic = brief.get("topic", "Тема презентации")
        outline = [
            {
                "title": f"{topic}: ключевой аспект {i}",
                "purpose": "Раскрыть тему для выбранной аудитории",
                "bullets": [
                    "Ключевая мысль слайда",
                    "Аргумент или пример",
                    "Практический вывод",
                ],
                "visual_hint": "Схема или сравнительная карточка",
                "speaker_notes": "Связать с целью презентации.",
                "body": (
                    "Этот слайд раскрывает один из ключевых аспектов темы и связывает "
                    "его с задачей презентации."
                ),
                "evidence": [],
                "takeaway": "Аудитория должна увидеть связь факта с общей идеей.",
            }
            for i in range(1, slides_count + 1)
        ]

    try:
        outline, enriched_closing = await _enrich_outline_with_llm(
            brief,
            outline,
            state.get("model", settings.LLM_MODEL),
        )
        closing_message = enriched_closing or closing_message
    except Exception:
        logger.exception("Presentation outline enrichment failed")

    outline = _repair_outline_content(outline, brief)
    if closing_message:
        brief = dict(brief)
        brief["closing_message"] = closing_message

    message = _render_outline(outline, brief)
    return {
        "brief": brief,
        "outline": outline,
        "stage": "outline_review",
        "brief_confirmed": False,
        "last_question_field": "",
        "status": message,
        "messages": [AIMessage(content=message)],
    }


async def generate_pptx_node(state: PresentationState) -> dict:
    """Создаёт PPTX из уже подтверждённой структуры."""
    from app.api.pptx_routes import (
        PresentationFromOutlineRequest,
        generate_presentation_from_outline,
    )

    brief = state.get("brief", {})
    outline = state.get("outline", [])

    if not outline:
        body = "Не вижу согласованной структуры. Сначала соберём и подтвердим план слайдов."
        return {
            "stage": "outline_review",
            "outline_confirmed": False,
            "status": body,
            "messages": [AIMessage(content=body)],
        }

    style = str(brief.get("style") or "corporate")
    request = PresentationFromOutlineRequest(
        brief=dict(brief),
        outline=outline,
        style=style,
        language=str(brief.get("language") or "ru"),
        user_id=state.get("user_id", "anonymous"),
    )

    try:
        response = await generate_presentation_from_outline(request)
        data = json.loads(response.body)
    except Exception as e:
        logger.exception("Presentation PPTX generation failed")
        body = f"Не удалось создать PPTX: {str(e)[:200]}"
        return {"status": body, "messages": [AIMessage(content=body)]}

    if "error" in data:
        body = f"Не удалось создать PPTX: {data['error']}"
        return {"status": body, "messages": [AIMessage(content=body)]}

    download_url = data["download_url"]
    slide_count = data["slide_count"]
    preview = data.get("preview_markdown", "")
    disclaimer = (
        "⚠️ **Важно:** презентация создана ИИ как рабочий черновик, а не как "
        "финальный материал для моментальной публикации. Крайне рекомендуется "
        "проверить факты, формулировки, оформление и при необходимости внести "
        "ручные правки перед показом или отправкой."
    )
    body = (
        f"Готово: презентация создана ({slide_count} слайдов).\n\n"
        f"{preview}\n\n"
        f"---\n"
        f"[Скачать PPTX](http://localhost:8000{download_url})\n\n"
        f"{disclaimer}"
    )
    return {
        "file_url": download_url,
        "stage": "generated",
        "status": body,
        "messages": [AIMessage(content=body)],
    }


def _route_after_extract(state: PresentationState) -> str:
    if state.get("outline_confirmed"):
        return "generate_pptx"
    if state.get("missing_fields"):
        return "ask_question"
    if state.get("brief_confirmed"):
        return "make_outline"
    return "show_brief"


def build_presentation_graph():
    graph = StateGraph(PresentationState)
    graph.add_node("extract_brief", extract_brief_node)
    graph.add_node("ask_question", ask_question_node)
    graph.add_node("show_brief", show_brief_node)
    graph.add_node("make_outline", make_outline_node)
    graph.add_node("generate_pptx", generate_pptx_node)

    graph.set_entry_point("extract_brief")
    graph.add_conditional_edges(
        "extract_brief",
        _route_after_extract,
        {
            "ask_question": "ask_question",
            "show_brief": "show_brief",
            "make_outline": "make_outline",
            "generate_pptx": "generate_pptx",
        },
    )
    graph.add_edge("ask_question", END)
    graph.add_edge("show_brief", END)
    graph.add_edge("make_outline", END)
    graph.add_edge("generate_pptx", END)

    return graph.compile()


presentation_graph = build_presentation_graph()


def _make_sse_chunk(content: str, model: str = "presentation-agent") -> str:
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


def _make_sse_finish(model: str = "presentation-agent") -> str:
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


def _dict_messages_to_lc(messages: list[dict]) -> list[BaseMessage]:
    lc_messages: list[BaseMessage] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            lc_messages.append(AIMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
    return lc_messages


def _last_ai_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return _stringify_content(msg.content)
    return ""


async def presentation_stream(
    messages: list[dict],
    user_id: str,
    session_id: str,
    model: str = "",
    memory_context: str = "",
) -> AsyncGenerator[str, None]:
    """
    Запускает интерактивный PPTX-граф и стримит один ответ через SSE.
    """
    sse_model = model or "presentation-agent"
    lc_messages = _dict_messages_to_lc(messages)
    last_user = _last_human_text(lc_messages)

    if _CANCEL_RE.match(last_user or ""):
        clear_presentation_session(session_id)
        yield _make_sse_chunk("Ок, остановил подготовку презентации.", sse_model)
        yield _make_sse_finish(sse_model)
        yield "data: [DONE]\n\n"
        return

    session = _get_session(session_id)

    initial_state: PresentationState = {
        "messages": lc_messages or [HumanMessage(content=last_user)],
        "user_id": user_id,
        "session_id": session_id,
        "model": model or settings.LLM_MODEL,
        "memory_context": memory_context,
        "last_user_text": last_user,
        "brief": dict(session.get("brief", {})),
        "missing_fields": [],
        "last_question_field": str(session.get("last_question_field", "")),
        "stage": session.get("stage", "collecting_brief"),
        "brief_confirmed": False,
        "outline": list(session.get("outline", [])),
        "outline_confirmed": False,
        "file_url": "",
        "status": "",
    }

    try:
        result = await presentation_graph.ainvoke(initial_state)
        _save_session(session_id, result)
        if result.get("stage") == "generated":
            clear_presentation_session(session_id)

        answer = _last_ai_text(result.get("messages", [])) or result.get("status", "")
        if not answer:
            answer = "Я продолжаю подготовку презентации. Уточни, пожалуйста, детали."
        yield _make_sse_chunk(answer, sse_model)
    except Exception as e:
        logger.exception("presentation_stream failed")
        yield _make_sse_chunk(f"Ошибка PPTX-агента: {str(e)[:300]}", sse_model)

    yield _make_sse_finish(sse_model)
    yield "data: [DONE]\n\n"
