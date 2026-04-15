"""
LangGraph агент для интерактивной подготовки Word-документов.

Задача графа — не генерировать DOCX сразу, а провести пользователя через
короткий брифинг:
  1. собрать тему, тип документа/стиль, аудиторию, цель и объём;
  2. уточнить материалы, факты, обязательные тезисы и ограничения;
  3. показать план документа;
  4. дождаться явного подтверждения;
  5. передать согласованный план в python-docx генератор.
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

from app.api.docx_routes import DOCUMENT_STYLE_SPECS, normalize_document_type
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger("mts.document_agent")


DocumentStage = Literal[
    "collecting_brief",
    "brief_review",
    "plan_review",
    "ready_to_generate",
    "generated",
]


class DocumentBrief(TypedDict, total=False):
    """Бриф Word-документа, постепенно собираемый из диалога."""

    topic: str
    document_type: str
    audience: str
    goal: str
    volume_pages: int
    language: str
    tone: str
    source_material: str
    key_facts: list[str]
    must_include: list[str]
    must_avoid: list[str]
    title: str
    subtitle: str


class DocumentPlanSection(TypedDict, total=False):
    """Раздел согласованного плана Word-документа."""

    heading: str
    level: int
    purpose: str
    key_points: list[str]
    facts_to_highlight: list[str]
    format_hint: str
    expected_elements: list[str]


class DocumentState(TypedDict):
    """Состояние интерактивного агента документов."""

    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    session_id: str
    model: str
    memory_context: str
    last_user_text: str

    brief: DocumentBrief
    missing_fields: list[str]
    last_question_field: str
    stage: DocumentStage
    brief_confirmed: bool

    plan: list[DocumentPlanSection]
    plan_confirmed: bool
    file_url: str
    status: str


REQUIRED_FIELDS: tuple[str, ...] = (
    "topic",
    "document_type",
    "audience",
    "goal",
    "volume_pages",
)

FIELD_LABELS: dict[str, str] = {
    "topic": "тема",
    "document_type": "тип документа / стиль Word",
    "audience": "адресат или аудитория",
    "goal": "цель документа",
    "volume_pages": "примерный объём",
}

DOCUMENT_TYPE_LABELS: dict[str, str] = {
    "official_business": "официальный деловой документ",
    "academic": "академическая работа",
    "business_report": "бизнес-документ / отчёт",
    "business_letter": "деловое письмо",
    "legal": "юридический документ",
    "marketing": "маркетинговый / презентационный документ",
    "informal_notes": "неформальные заметки",
    "technical_docs": "техническая документация",
    "instruction": "инструкция / руководство",
}

_DOCUMENT_SESSIONS: dict[str, dict[str, Any]] = {}
_SESSION_TTL_SECONDS = 60 * 60 * 3

_CONFIRM_RE = re.compile(
    r"^\s*(да|ок|окей|подтверждаю|утверждаю|всё верно|все верно|"
    r"поехали|давай|начинай|старт|го|go|делай|создавай|генерируй|можно создавать|"
    r"план подходит|подходит|согласен)\s*[.!?]*\s*$",
    re.IGNORECASE,
)

_CANCEL_RE = re.compile(
    r"^\s*(отмена|отмени|стоп|stop|cancel|не надо|закрой режим)\s*[.!?]*\s*$",
    re.IGNORECASE,
)

_NEGATIVE_RE = re.compile(
    r"\b(измени|поменяй|добавь|убери|замени|не подходит|не то|лучше|"
    r"переделай|пересобери|исправь|дополни)\b",
    re.IGNORECASE,
)


BRIEF_EXTRACT_PROMPT = """\
Ты — эксперт по подготовке Word-документов и редактор деловой письменности.

Извлеки из диалога параметры будущего DOCX-документа. Не придумывай значения,
если пользователь их не сообщил явно.

Текущий бриф, если он уже есть:
{brief_json}

Диалог:
{transcript}

Верни ТОЛЬКО валидный JSON без markdown:
{{
  "topic": string|null,
  "document_type": string|null,
  "audience": string|null,
  "goal": string|null,
  "volume_pages": integer|null,
  "language": "ru"|"en"|null,
  "tone": string|null,
  "source_material": string|null,
  "key_facts": string[],
  "must_include": string[],
  "must_avoid": string[],
  "title": string|null,
  "subtitle": string|null
}}

Правила нормализации document_type:
- "official_business" — официальный деловой документ, ГОСТ, служебный документ, заявление, приказ.
- "academic" — реферат, курсовая, диплом, научный доклад, академическая работа.
- "business_report" — бизнес-отчёт, предложение, аналитическая записка, executive summary.
- "business_letter" — деловое письмо, email, обращение кому-то.
- "legal" — договор, соглашение, юридический документ, оферта, правовые условия.
- "marketing" — маркетинговый, продающий, коммерческий, презентационный документ.
- "informal_notes" — неформальные заметки, свободный черновик.
- "technical_docs" — техническая документация, ТЗ, API, архитектура, требования.
- "instruction" — инструкция, руководство, пошаговый гайд.
- Если пользователь просит "Word", "docx" или "документ", но тип неясен, document_type = null.

Дополнительные правила:
- Если пользователь прислал большой фрагмент текста/заметок/сырой материал, сохрани его смысл в source_material.
- key_facts — только явно предоставленные факты или тезисы, которые надо выделить.
- must_include — то, что пользователь явно просит включить.
- must_avoid — то, что пользователь явно просит не использовать.
- Если объём не указан, volume_pages = null.
- Если язык не указан, language = "ru".
- Никакого текста вне JSON.
"""

PLAN_PROMPT = """\
Ты — senior document architect. Составь сильный план Word-документа перед генерацией.

Бриф:
{brief_json}

Контекст из памяти пользователя:
{memory_context}

Стандарт оформления выбранного стиля:
{style_json}

{revision_block}

Верни ТОЛЬКО валидный JSON без markdown:
{{
  "document_title": "...",
  "subtitle": "...",
  "sections": [
    {{
      "heading": "...",
      "level": 1,
      "purpose": "зачем этот раздел нужен",
      "key_points": ["смысловой тезис", "аргумент", "вывод"],
      "facts_to_highlight": ["конкретный факт из материала или пустой список"],
      "format_hint": "paragraphs|bullets|numbered_list|table|code|callout",
      "expected_elements": ["paragraphs", "table"]
    }}
  ]
}}

Жёсткие правила:
- План должен соответствовать типу документа и стандартам оформления, а не быть универсальной болванкой.
- Если document_type = official_business: строгая структура, формальные формулировки, нумерация 1 / 1.1 / 1.1.1.
- Если academic: титульный лист, содержание, введение, главы, заключение, список литературы.
- Если business_report: executive summary, контекст, анализ, выводы, рекомендации.
- Если business_letter: кому/от кого/дата, обращение, суть, завершение, подпись.
- Если legal: предмет, термины, права и обязанности, порядок действия, ответственность, подписи.
- Если marketing: оффер, выгоды, доказательства, возражения, следующий шаг.
- Если technical_docs: описание, требования, инструкции, примеры, ограничения.
- Если instruction: перед началом, пошаговые действия, проверка результата, ошибки.
- Каждый раздел должен иметь конкретную роль и вывод, а не пустой заголовок.
- Если есть source_material, план должен разложить его факты по разделам.
- Не выдумывай факты, даты, цифры, источники, законы и названия организаций.
- Если факта не хватает, запланируй аккуратную формулировку "требует проверки".
- document_title: короткое редакторское название, не более 12 слов.
- Количество разделов подбери под brief.volume_pages: 1-2 страницы = 3-5 разделов, 3-6 страниц = 5-8 разделов, больше = 8-12 разделов.
"""


def _now() -> float:
    return time.time()


def _purge_expired_sessions() -> None:
    cutoff = _now() - _SESSION_TTL_SECONDS
    expired = [
        key
        for key, value in _DOCUMENT_SESSIONS.items()
        if value.get("updated_at", 0) < cutoff
    ]
    for key in expired:
        _DOCUMENT_SESSIONS.pop(key, None)


def has_active_document_session(session_id: str) -> bool:
    """Есть ли незавершённый DOCX-брифинг для этого чата."""
    _purge_expired_sessions()
    session = _DOCUMENT_SESSIONS.get(session_id)
    if not session:
        return False
    return session.get("stage") != "generated"


def clear_document_session(session_id: str) -> None:
    """Сбрасывает активный DOCX-брифинг."""
    _DOCUMENT_SESSIONS.pop(session_id, None)


def _get_session(session_id: str) -> dict[str, Any]:
    _purge_expired_sessions()
    session = _DOCUMENT_SESSIONS.get(session_id)
    if not session:
        session = {
            "brief": {},
            "plan": [],
            "stage": "collecting_brief",
            "last_question_field": "",
            "updated_at": _now(),
        }
        _DOCUMENT_SESSIONS[session_id] = session
    return session


def _save_session(session_id: str, state: DocumentState) -> None:
    _DOCUMENT_SESSIONS[session_id] = {
        "brief": dict(state.get("brief", {})),
        "plan": list(state.get("plan", [])),
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
        logger.warning("Document JSON parse failed: %r", raw[:300])
        return {}
    return obj if isinstance(obj, dict) else {}


def _shorten(text: Any, limit: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    cut = value[: max(0, limit - 1)].rstrip()
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(".,;:") + "..."


def _normalize_brief(raw: dict[str, Any]) -> DocumentBrief:
    brief: DocumentBrief = {}

    for key in ("topic", "audience", "goal", "language", "tone", "source_material", "title", "subtitle"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            brief[key] = value.strip()

    document_type = raw.get("document_type")
    if document_type:
        normalized = normalize_document_type(document_type)
        if normalized:
            brief["document_type"] = normalized

    volume = raw.get("volume_pages")
    if isinstance(volume, str):
        match = re.search(r"\d+", volume)
        volume = int(match.group()) if match else None
    if isinstance(volume, (int, float)):
        brief["volume_pages"] = max(1, min(int(volume), 25))

    for key in ("key_facts", "must_include", "must_avoid"):
        value = raw.get(key)
        if isinstance(value, list):
            items = [str(v).strip() for v in value if str(v).strip()]
            if items:
                brief[key] = items[:12]

    if "language" not in brief:
        brief["language"] = "ru"

    return brief


def _merge_brief(old: DocumentBrief, new: DocumentBrief) -> DocumentBrief:
    merged: DocumentBrief = dict(old)
    for key, value in new.items():
        if value not in (None, "", [], {}):
            merged[key] = value
    return merged


def _missing_fields(brief: DocumentBrief) -> list[str]:
    missing: list[str] = []
    for field in REQUIRED_FIELDS:
        value = brief.get(field)
        if value in (None, "", [], {}):
            missing.append(field)
    return missing


def _extract_topic_from_request(text: str) -> str:
    t = text.strip()
    t = re.sub(
        r"^\s*(сделай|создай|подготовь|сгенерируй|собери|напиши|оформи)\s+",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"\b(word|docx|ворд|документ|файл|отч[её]т|письмо|инструкци[яю]|руководство)\b",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"\b(на тему|по теме|про|о|в формате)\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip(" .,:;-")
    return t


def _fallback_extract(text: str, current_brief: DocumentBrief) -> DocumentBrief:
    """Детерминированный минимум, если LLM недоступна или дала плохой JSON."""
    extracted: DocumentBrief = {}
    t = text.lower()

    if "topic" not in current_brief and re.search(r"word|docx|ворд|документ|отч[её]т|письм|инструкц", t):
        topic = _extract_topic_from_request(text)
        if len(topic) >= 3:
            extracted["topic"] = topic

    detected_type = normalize_document_type(text)
    if detected_type != "business_report" or re.search(r"бизнес|отч[её]т|proposal|предлож", t):
        extracted["document_type"] = detected_type

    pages_match = re.search(r"(\d{1,2})\s*(?:стр|страниц|page)", t)
    if pages_match:
        extracted["volume_pages"] = max(1, min(int(pages_match.group(1)), 25))

    audience_match = re.search(r"\b(?:для|кому|адресат[:\s-]+)\s+([^.,;!?]+)", text, re.IGNORECASE)
    if audience_match:
        audience = audience_match.group(1).strip()
        if 3 <= len(audience) <= 160:
            extracted["audience"] = audience

    goal_match = re.search(r"\bцель[:\s-]+([^.;!?]+)", text, re.IGNORECASE)
    if goal_match:
        goal = goal_match.group(1).strip()
        if 3 <= len(goal) <= 220:
            extracted["goal"] = goal

    if len(text) > 160 and not re.search(r"^\s*(сделай|создай|подготовь|сгенерируй|оформи|напиши)", t):
        extracted["source_material"] = text[:9000].strip()

    extracted["language"] = "ru"
    return extracted


def _coerce_answer_for_field(field: str, text: str) -> DocumentBrief:
    clean = text.strip(" \n\t.,;")
    if not clean or _CANCEL_RE.match(clean):
        return {}
    if len(clean) > 600:
        clean = clean[:600].strip()

    if field == "volume_pages":
        match = re.search(r"\d{1,2}", clean)
        if match:
            return {"volume_pages": max(1, min(int(match.group()), 25))}
        return {}

    if field == "document_type":
        document_type = normalize_document_type(clean)
        return {"document_type": document_type} if document_type else {}

    if field in {"topic", "audience", "goal"} and len(clean) >= 2:
        return {field: clean}

    return {}


async def _extract_brief_with_llm(
    messages: list[BaseMessage],
    current_brief: DocumentBrief,
    model: str,
) -> DocumentBrief:
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


def _is_confirmation(text: str) -> bool:
    clean = text.strip()
    if not clean:
        return False
    if _NEGATIVE_RE.search(clean):
        return False
    return bool(_CONFIRM_RE.match(clean))


def _make_question(field: str, brief: DocumentBrief) -> str:
    topic = brief.get("topic", "документа")
    templates = {
        "topic": (
            "Давай соберём нормальный Word-бриф. Какая точная тема документа? "
            "Если есть сырой текст, факты или заметки, присылай сюда — я разложу их по разделам."
        ),
        "document_type": (
            "Какой тип Word-документа нужен? Например: **официальный деловой/ГОСТ**, "
            "**академическая работа**, бизнес-отчёт, деловое письмо, юридический документ, "
            "маркетинговый текст, техническая документация или инструкция."
        ),
        "audience": (
            f"Понял тему: **{topic}**. Для кого или кому готовим документ: руководство, "
            "клиент, преподаватель, юристы, техкоманда, сотрудники или другая аудитория?"
        ),
        "goal": (
            "Какую цель должен решить документ: проинформировать, убедить, зафиксировать "
            "договорённости, обучить, защитить работу, продать идею или что-то другое?"
        ),
        "volume_pages": (
            "Какой примерный объём нужен в страницах Word? Можно ответить числом, например: **3**. "
            "Если фактов мало, я аккуратно отмечу, где нужна проверка."
        ),
    }
    return templates.get(field, f"Уточни, пожалуйста: {FIELD_LABELS.get(field, field)}?")


def _normalize_plan(raw: dict[str, Any], brief: DocumentBrief) -> list[DocumentPlanSection]:
    sections_raw = raw.get("sections", [])
    if not isinstance(sections_raw, list):
        return []

    sections: list[DocumentPlanSection] = []
    for idx, item in enumerate(sections_raw, 1):
        if not isinstance(item, dict):
            continue
        sections.append(
            {
                "heading": _shorten(item.get("heading") or f"Раздел {idx}", 120),
                "level": max(1, min(int(item.get("level") or 1), 3)),
                "purpose": _shorten(item.get("purpose") or "", 220),
                "key_points": [
                    _shorten(point, 220)
                    for point in item.get("key_points", [])
                    if str(point).strip()
                ][:8]
                if isinstance(item.get("key_points", []), list)
                else [],
                "facts_to_highlight": [
                    _shorten(fact, 220)
                    for fact in item.get("facts_to_highlight", [])
                    if str(fact).strip()
                ][:8]
                if isinstance(item.get("facts_to_highlight", []), list)
                else [],
                "format_hint": _shorten(item.get("format_hint") or "", 120),
                "expected_elements": [
                    str(element).strip()
                    for element in item.get("expected_elements", [])
                    if str(element).strip()
                ][:5]
                if isinstance(item.get("expected_elements", []), list)
                else [],
            }
        )

    volume = int(brief.get("volume_pages", 3) or 3)
    max_sections = 5 if volume <= 2 else 8 if volume <= 6 else 12
    return sections[:max_sections]


def _repair_plan(plan: list[DocumentPlanSection], brief: DocumentBrief) -> list[DocumentPlanSection]:
    if plan:
        repaired: list[DocumentPlanSection] = []
        for idx, section in enumerate(plan, 1):
            item = dict(section)
            if not item.get("purpose"):
                item["purpose"] = f"Раскрыть роль раздела «{item.get('heading', idx)}» в логике документа."
            if not item.get("key_points"):
                item["key_points"] = [
                    f"Показать связь раздела с темой: {brief.get('topic', 'документ')}",
                    f"Подвести к цели: {brief.get('goal', 'донести ключевой смысл')}",
                ]
            if not item.get("format_hint"):
                item["format_hint"] = "paragraphs"
            if not item.get("expected_elements"):
                item["expected_elements"] = ["paragraphs"]
            repaired.append(item)
        return repaired

    document_type = str(brief.get("document_type") or "business_report")
    defaults = {
        "official_business": ["Основание", "Основные положения", "Порядок исполнения", "Заключительные положения"],
        "academic": ["Введение", "Глава 1. Теоретические основы", "Глава 2. Анализ", "Заключение", "Список литературы"],
        "business_report": ["Executive Summary", "Контекст", "Анализ", "Выводы", "Рекомендации"],
        "business_letter": ["Реквизиты и обращение", "Суть письма", "Ожидаемое действие", "Подпись"],
        "legal": ["Предмет документа", "Права и обязанности", "Порядок действия", "Ответственность", "Подписи"],
        "marketing": ["Ключевое предложение", "Выгоды", "Доказательства", "Работа с возражениями", "Следующий шаг"],
        "informal_notes": ["Контекст", "Основные заметки", "Что важно запомнить", "Следующие шаги"],
        "technical_docs": ["Описание", "Требования", "Порядок работы", "Примеры", "Ограничения"],
        "instruction": ["Перед началом", "Пошаговые действия", "Проверка результата", "Частые ошибки"],
    }
    headings = defaults.get(document_type, defaults["business_report"])
    return [
        {
            "heading": heading,
            "level": 1,
            "purpose": f"Раскрыть раздел «{heading}» применительно к теме документа.",
            "key_points": [str(brief.get("topic") or "Тема документа")],
            "facts_to_highlight": list(brief.get("key_facts", []))[:3],
            "format_hint": "paragraphs",
            "expected_elements": ["paragraphs", "bullets"],
        }
        for heading in headings
    ]


def _render_brief_review(brief: DocumentBrief) -> str:
    document_type = str(brief.get("document_type") or "")
    style_label = DOCUMENT_TYPE_LABELS.get(document_type, document_type or "—")
    lines = [
        "Бриф для Word-документа собран. Проверь, всё ли верно:",
        "",
        f"**Тема:** {brief.get('topic', '—')}",
        f"**Тип/стиль:** {style_label}",
        f"**Аудитория/адресат:** {brief.get('audience', '—')}",
        f"**Цель:** {brief.get('goal', '—')}",
        f"**Объём:** {brief.get('volume_pages', '—')} стр.",
    ]
    if brief.get("tone"):
        lines.append(f"**Тон:** {brief['tone']}")
    if brief.get("key_facts"):
        lines.append("**Факты для выделения:** " + "; ".join(brief["key_facts"]))
    if brief.get("must_include"):
        lines.append("**Обязательно включить:** " + "; ".join(brief["must_include"]))
    if brief.get("must_avoid"):
        lines.append("**Не использовать:** " + "; ".join(brief["must_avoid"]))
    if brief.get("source_material"):
        source = str(brief["source_material"])
        preview = source[:320] + ("..." if len(source) > 320 else "")
        lines.append(f"**Исходный материал:** {preview}")

    lines += [
        "",
        "Если бриф верный, напиши **«поехали»** — я составлю план документа.",
        "Можно также докинуть факты, требования к структуре или пример желаемого стиля.",
        "Если нужно поправить вводные, напиши правку обычным текстом.",
    ]
    return "\n".join(lines)


def _render_plan(plan: list[DocumentPlanSection], brief: DocumentBrief) -> str:
    document_type = str(brief.get("document_type") or "")
    style_label = DOCUMENT_TYPE_LABELS.get(document_type, document_type or "—")
    spec = DOCUMENT_STYLE_SPECS.get(document_type, {})
    lines = [
        "Я собрал план Word-документа. Проверь, пожалуйста:",
        "",
        f"**Тема:** {brief.get('topic', '—')}",
        f"**Стиль:** {style_label}",
    ]
    if spec:
        lines.extend(
            [
                f"**Оформление:** {spec.get('font')} {spec.get('font_size')} pt, "
                f"интервал {spec.get('line_spacing')}, поля: "
                f"левое {spec.get('margins_cm', {}).get('left')} см, "
                f"правое {spec.get('margins_cm', {}).get('right')} см",
                f"**Тон:** {spec.get('tone')}",
                "",
            ]
        )

    for idx, section in enumerate(plan, 1):
        prefix = f"{idx}." if int(section.get("level", 1)) == 1 else f"{idx}"
        lines.append(f"**{prefix} {section.get('heading', f'Раздел {idx}')}**")
        if section.get("purpose"):
            lines.append(f"Задача: {section['purpose']}")
        for point in (section.get("key_points") or [])[:3]:
            lines.append(f"- {point}")
        facts = section.get("facts_to_highlight") or []
        if facts:
            lines.append("Факты: " + "; ".join(facts[:3]))
        if section.get("format_hint"):
            lines.append(f"Формат: {section['format_hint']}")
        lines.append("")

    lines.append(
        "Если всё подходит, напиши **«подтверждаю»** — после этого я создам DOCX. "
        "Если нужно поправить план, напиши, что изменить."
    )
    return "\n".join(lines).strip()


async def extract_brief_node(state: DocumentState) -> dict:
    """Обновляет бриф по текущей реплике пользователя и истории диалога."""
    last_user = state.get("last_user_text") or _last_human_text(state["messages"])
    brief = dict(state.get("brief", {}))
    stage = state.get("stage", "collecting_brief")

    if stage == "plan_review" and state.get("plan"):
        confirmed = _is_confirmation(last_user)
        if confirmed:
            return {
                "plan_confirmed": True,
                "missing_fields": [],
                "stage": "ready_to_generate",
                "status": "План подтверждён, создаю DOCX...",
            }
        if last_user:
            source_update = {}
            if len(last_user) > 80:
                previous_source = str(brief.get("source_material") or "")
                source_update["source_material"] = (
                    previous_source + "\n\n" + last_user
                ).strip()[-12000:]
            brief = _merge_brief(brief, source_update)
            return {
                "brief": brief,
                "brief_confirmed": True,
                "plan_confirmed": False,
                "missing_fields": [],
                "last_question_field": "",
                "stage": "plan_review",
                "status": "Пересобираю план документа с учётом правок...",
            }

    if stage == "brief_review" and _is_confirmation(last_user):
        return {
            "brief_confirmed": True,
            "plan_confirmed": False,
            "missing_fields": [],
            "last_question_field": "",
            "stage": "plan_review",
            "status": "Бриф подтверждён, собираю план документа...",
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
        logger.exception("Document brief extraction via LLM failed")
        llm_update = {}

    fallback_update = _fallback_extract(last_user, brief)
    brief = _merge_brief(brief, fallback_update)
    brief = _merge_brief(brief, llm_update)
    brief = _merge_brief(brief, direct_update)

    missing = _missing_fields(brief)
    next_stage: DocumentStage = "collecting_brief" if missing else "brief_review"

    return {
        "brief": brief,
        "missing_fields": missing,
        "brief_confirmed": False,
        "plan_confirmed": False,
        "stage": next_stage,
    }


async def ask_question_node(state: DocumentState) -> dict:
    """Задаёт один следующий вопрос, без генерации плана."""
    missing = state.get("missing_fields", [])
    field = missing[0] if missing else ""
    question = _make_question(field, state.get("brief", {}))
    return {
        "last_question_field": field,
        "status": question,
        "messages": [AIMessage(content=question)],
    }


async def show_brief_node(state: DocumentState) -> dict:
    """Показывает собранный бриф и ждёт отмашку на создание плана."""
    message = _render_brief_review(state.get("brief", {}))
    return {
        "stage": "brief_review",
        "last_question_field": "",
        "status": message,
        "messages": [AIMessage(content=message)],
    }


async def make_plan_node(state: DocumentState) -> dict:
    """Генерирует или пересобирает план и отдаёт его на подтверждение."""
    brief = dict(state.get("brief", {}))
    last_user = state.get("last_user_text", "")
    current_plan = state.get("plan", [])
    document_type = str(brief.get("document_type") or "business_report")
    spec = DOCUMENT_STYLE_SPECS.get(document_type, DOCUMENT_STYLE_SPECS["business_report"])

    revision_block = ""
    if current_plan and last_user and not _is_confirmation(last_user):
        revision_block = (
            "Текущий план уже был предложен:\n"
            f"{json.dumps(current_plan, ensure_ascii=False)}\n\n"
            "Пользователь попросил изменить его так:\n"
            f"{last_user}\n\n"
            "Пересобери план с учётом правок."
        )

    prompt = PLAN_PROMPT.format(
        brief_json=json.dumps(brief, ensure_ascii=False),
        memory_context=state.get("memory_context", "") or "нет",
        style_json=json.dumps(spec, ensure_ascii=False),
        revision_block=revision_block,
    )

    llm = ChatOpenAI(
        base_url=settings.MWS_BASE_URL,
        api_key=settings.MWS_API_KEY,
        model=state.get("model", settings.LLM_MODEL),
        temperature=0.25,
        streaming=False,
    )

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        obj = _extract_json_object(str(response.content))
        plan = _normalize_plan(obj, brief)
        if obj.get("document_title"):
            brief["title"] = _shorten(obj.get("document_title"), 120)
        if obj.get("subtitle"):
            brief["subtitle"] = _shorten(obj.get("subtitle"), 160)
    except Exception:
        logger.exception("Document plan generation failed")
        plan = []

    plan = _repair_plan(plan, brief)
    message = _render_plan(plan, brief)
    return {
        "brief": brief,
        "plan": plan,
        "stage": "plan_review",
        "brief_confirmed": False,
        "last_question_field": "",
        "status": message,
        "messages": [AIMessage(content=message)],
    }


async def generate_docx_node(state: DocumentState) -> dict:
    """Создаёт DOCX из уже подтверждённого плана."""
    from app.api.docx_routes import DocumentFromPlanRequest, generate_document_from_plan

    brief = state.get("brief", {})
    plan = state.get("plan", [])

    if not plan:
        body = "Не вижу согласованного плана. Сначала соберём и подтвердим структуру документа."
        return {
            "stage": "plan_review",
            "plan_confirmed": False,
            "status": body,
            "messages": [AIMessage(content=body)],
        }

    document_type = str(brief.get("document_type") or "business_report")
    request = DocumentFromPlanRequest(
        brief=dict(brief),
        plan=plan,
        document_type=document_type,
        language=str(brief.get("language") or "ru"),
        user_id=state.get("user_id", "anonymous"),
    )

    try:
        response = await generate_document_from_plan(request)
        data = json.loads(response.body)
    except Exception as e:
        logger.exception("DOCX generation failed")
        body = f"Не удалось создать DOCX: {str(e)[:200]}"
        return {"status": body, "messages": [AIMessage(content=body)]}

    if "error" in data:
        body = f"Не удалось создать DOCX: {data['error']}"
        return {"status": body, "messages": [AIMessage(content=body)]}

    download_url = data["download_url"]
    section_count = data["section_count"]
    preview = data.get("preview_markdown", "")
    disclaimer = (
        "⚠️ **Важно:** документ создан ИИ как рабочий черновик, а не как "
        "финальный материал для моментальной публикации или юридически значимого "
        "использования. Крайне рекомендуется проверить факты, формулировки, "
        "оформление, источники и при необходимости внести ручные правки перед "
        "отправкой или подписанием."
    )
    body = (
        f"Готово: Word-документ создан ({section_count} разделов).\n\n"
        f"{preview}\n\n"
        f"---\n"
        f"[Скачать DOCX](http://localhost:8000{download_url})\n\n"
        f"{disclaimer}"
    )
    return {
        "file_url": download_url,
        "stage": "generated",
        "status": body,
        "messages": [AIMessage(content=body)],
    }


def _route_after_extract(state: DocumentState) -> str:
    if state.get("plan_confirmed"):
        return "generate_docx"
    if state.get("missing_fields"):
        return "ask_question"
    if state.get("brief_confirmed"):
        return "make_plan"
    return "show_brief"


def build_document_graph():
    graph = StateGraph(DocumentState)
    graph.add_node("extract_brief", extract_brief_node)
    graph.add_node("ask_question", ask_question_node)
    graph.add_node("show_brief", show_brief_node)
    graph.add_node("make_plan", make_plan_node)
    graph.add_node("generate_docx", generate_docx_node)

    graph.set_entry_point("extract_brief")
    graph.add_conditional_edges(
        "extract_brief",
        _route_after_extract,
        {
            "ask_question": "ask_question",
            "show_brief": "show_brief",
            "make_plan": "make_plan",
            "generate_docx": "generate_docx",
        },
    )
    graph.add_edge("ask_question", END)
    graph.add_edge("show_brief", END)
    graph.add_edge("make_plan", END)
    graph.add_edge("generate_docx", END)

    return graph.compile()


document_graph = build_document_graph()


def _make_sse_chunk(content: str, model: str = "document-agent") -> str:
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


def _make_sse_finish(model: str = "document-agent") -> str:
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


async def document_stream(
    messages: list[dict],
    user_id: str,
    session_id: str,
    model: str = "",
    memory_context: str = "",
) -> AsyncGenerator[str, None]:
    """
    Запускает интерактивный DOCX-граф и стримит один ответ через SSE.
    """
    sse_model = model or "document-agent"
    lc_messages = _dict_messages_to_lc(messages)
    last_user = _last_human_text(lc_messages)

    if _CANCEL_RE.match(last_user or ""):
        clear_document_session(session_id)
        yield _make_sse_chunk("Ок, остановил подготовку Word-документа.", sse_model)
        yield _make_sse_finish(sse_model)
        yield "data: [DONE]\n\n"
        return

    session = _get_session(session_id)

    initial_state: DocumentState = {
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
        "plan": list(session.get("plan", [])),
        "plan_confirmed": False,
        "file_url": "",
        "status": "",
    }

    try:
        result = await document_graph.ainvoke(initial_state)
        _save_session(session_id, result)
        if result.get("stage") == "generated":
            clear_document_session(session_id)

        answer = _last_ai_text(result.get("messages", [])) or result.get("status", "")
        if not answer:
            answer = "Я продолжаю подготовку Word-документа. Уточни, пожалуйста, детали."
        yield _make_sse_chunk(answer, sse_model)
    except Exception as e:
        logger.exception("document_stream failed")
        yield _make_sse_chunk(f"Ошибка DOCX-агента: {str(e)[:300]}", sse_model)

    yield _make_sse_finish(sse_model)
    yield "data: [DONE]\n\n"
