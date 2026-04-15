"""
LangGraph граф для режима «🔬 Deep Research».

Многошаговый цикл исследования:
  1. Planner    — разбивает запрос на подвопросы
  2. Researcher — для каждого подвопроса: поиск → парсинг → извлечение фактов
  3. Reviewer   — проверяет: ещё исследовать или писать отчёт
  4. Writer     — синтезирует финальный отчёт из всех находок

Использование:
    from app.graph.deep_research import deep_research_stream
    async for sse_chunk in deep_research_stream(query, user_id, ...):
        ...
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
import hashlib
from pathlib import Path
from typing import Annotated, Any, AsyncGenerator
from datetime import datetime

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.core.config import get_settings
from app.core import prompts
from app.services import web_tools

settings = get_settings()
logger = logging.getLogger("mts.deep_research")


class ResearchState(TypedDict):
    """Состояние Deep Research графа."""

    messages: Annotated[list[BaseMessage], add_messages]
    query: str                    # Исходный запрос пользователя
    sub_questions: list[str]      # Сгенерированные подвопросы
    current_idx: int              # Индекс текущего подвопроса
    findings: list[str]           # Найденные факты по каждому подвопросу
    report: str                   # Финальный отчёт
    status: str                   # Текущий статус для SSE
    model: str                    # Модель LLM
    user_id: str
    memory_context: str
    max_questions: int            # Лимит подвопросов (настраиваемый)
    max_urls: int                 # Лимит URL на подвопрос (настраиваемый)
    iteration: int                # Текущая итерация цикла
    plan_confirmed: bool          # Подтвердил ли пользователь план
    needs_confirmation: bool      # Нужно ли выдать план и остановиться
    last_reflection: str          # Последнее осмысление
    retry_count: int              # Счётчик попыток для подвопроса
    is_ambiguous: bool            # Запрос слишком широкий/неясный
    clarification_options: list[str] # Варианты уточнения
    briefing_questions: list[str] # Вопросы для брифинга
    briefing_answers: str         # Ответы на брифинг


# Helpers

# Множество слов-подтверждений для быстрого поиска (без substring-матча!)
_CONFIRM_WORDS = frozenset([
    "поехали", "начинай", "ок", "подтверждаю", "давай", "старт", "гоу",
    "всё верно", "продолжай", "да", "yes", "go", "start",
    "1", "2", "3", "первый", "второй", "третий",
])


def _is_pure_confirmation(text: str) -> bool:
    """
    True только если сообщение ≤ 3 слова и ВСЕ слова — слова-подтверждения.
    Исправляет P1/P4/P7: 'давай повторно исследование сделаем' содержит 'давай',
    но НЕ является подтверждением — это новый контекстный запрос.
    """
    words = text.lower().strip().split()
    if len(words) > 3:
        return False
    return all(w in _CONFIRM_WORDS for w in words)


def _get_selected_option_idx(text: str) -> int | None:
    """Если пользователь написал 1/2/3 — возвращает 0-based индекс выбора."""
    t = text.lower().strip()
    mapping = {"1": 0, "первый": 0, "2": 1, "второй": 1, "3": 2, "третий": 2}
    return mapping.get(t)


def _get_llm(model: str | None = None) -> ChatOpenAI:
    """Создаёт экземпляр LLM для использования в нодах."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        base_url=settings.MWS_BASE_URL,
        api_key=settings.MWS_API_KEY,
        model=model or settings.LLM_MODEL,
        temperature=0.3,  # Низкая для точности исследования
        streaming=False,
    )


def _clean_llm_output(text: str) -> str:
    """Унифицированная очистка вывода (теперь сохраняет <think> для UI)."""
    if not text:
        return ""
    # Очищаем только Markdown артефакты если нужно, но НЕ <think> блоки
    return text.strip()


async def update_planner_md(state: ResearchState, node_name: str) -> None:
    """Обновляет файл research_plan.md в кеше для визуализации прогресса."""
    user_id = state.get("user_id", "unknown")
    query = state.get("query", "No query")
    sub_questions = state.get("sub_questions", [])
    current_idx = state.get("current_idx", 0)
    findings = state.get("findings", [])
    report = state.get("report", "")
    
    # Путь к файлу лога для данного пользователя
    log_dir = Path("app/data/research_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    plan_file = log_dir / f"plan_{user_id}.md"

    md_content = f"# 🔬 Исследование: {query}\n\n"
    
    if state.get("is_ambiguous"):
        md_content += "⚠️ **Статус: Ожидание уточнения запроса**\n\n"
        if state.get("clarification_options"):
            md_content += "Нужно выбрать один из вариантов:\n"
            for opt in state["clarification_options"]:
                md_content += f"- {opt}\n"
    elif state.get("briefing_questions") and not state.get("briefing_answers"):
        md_content += "📋 **Статус: Бриф (сбор доп. информации)**\n\n"
        md_content += "Агент ожидает ответы на вопросы:\n"
        for q in state["briefing_questions"]:
            md_content += f"- {q}\n"
    else:
        status_icon = "✅" if report else "⚙️"
        md_content += f"## {status_icon} Статус: {'Готово' if report else 'В процессе'}\n\n"
        
        md_content += "### 📋 План задач\n"
        if not sub_questions:
            md_content += "*План еще составляется...*\n"
        for i, q in enumerate(sub_questions):
            if i < current_idx:
                md_content += f"- [x] {q}\n"
            elif i == current_idx and not report:
                md_content += f"- [ ] **{q}** (Исследую...)\n"
            else:
                md_content += f"- [ ] {q}\n"
        
        if findings:
            md_content += "\n### 📝 Промежуточные результаты\n"
            for f in findings:
                # Берем только заголовок находки
                title = f.split("\n")[0].replace("### ", "")
                md_content += f"- {title}\n"
        
        if report:
            md_content += "\n---\n## 📊 Финальный отчет подготовлен"

    try:
        with open(plan_file, "w", encoding="utf-8") as f:
            f.write(md_content)
        # logger.info("📝 Updated MD plan at %s (Node: %s)", plan_file, node_name)
    except Exception as e:
        logger.error("Error writing MD plan: %s", e)


# Graph nodes


async def clarify_node(state: ResearchState) -> dict:
    """Проверяет запрос на однозначность и полноту."""
    query = state["query"]
    history = state.get("messages", [])
    
    # P2: Если пользователь выбирает вариант 1/2/3 из предыдущего clarify — ставим выбранную тему
    sel_idx = _get_selected_option_idx(query)
    if sel_idx is not None:
        for m in reversed(history):
            if isinstance(m, AIMessage) and "Пожалуйста, уточните" in m.content:
                options = []
                for line in m.content.split("\n"):
                    opt_match = re.match(r"^\s*\d+\.\s*(.*)", line)
                    if opt_match:
                        options.append(opt_match.group(1).strip())
                if options and sel_idx < len(options):
                    chosen = options[sel_idx]
                    logger.info("🤔 [CLARIFY] User selected option %d: %s", sel_idx + 1, chosen)
                    return {"is_ambiguous": False, "query": chosen}
                break

    # P1: Используем точную проверку (не substring!) — чистое подтверждение ≤ 3 слов
    if _is_pure_confirmation(query):
        return {"is_ambiguous": False}

    prompt = prompts.research_clarify_prompt.get_prompt(query=query)
    
    llm = _get_llm(state.get("model"))
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = _clean_llm_output(response.content)
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            res = json.loads(match.group())
            is_ambiguous = res.get("is_ambiguous", False)
            options = res.get("options", [])
            
            if is_ambiguous and options:
                status = f"🤔 **Запрос кажется слишком широким.** Пожалуйста, уточните, что именно мы исследуем:\n\n"
                for i, opt in enumerate(options, 1):
                    status += f" {i}. {opt}\n"
                status += "\nВыберите вариант или введите свой запрос точнее."
                
                new_state = {
                    "is_ambiguous": True,
                    "clarification_options": options,
                    "status": status,
                    "needs_confirmation": True,
                    "messages": [AIMessage(content=status)]
                }
                # P14: передаём полный state, чтобы plan_{user_id}.md не был пустым
                await update_planner_md({**state, **new_state}, "clarify")
                return new_state
    except Exception as e:
        logger.error("Error in clarify_node: %s", e)

    return {"is_ambiguous": False}


async def brief_node(state: ResearchState) -> dict:
    """Генерирует вопросы для уточнения деталей сложного исследования (брифинг)."""
    query = state["query"]
    history = state.get("messages", [])
    
    # Ищем бриф в истории и проверяем, что он ОТНОСИТСЯ к текущему запросу
    brief_idx = None
    for i, m in enumerate(history):
        if isinstance(m, AIMessage) and "Бриф исследования" in m.content:
            brief_idx = i

    # Проверяем релевантность: если бриф из прошлой сессии (тема не совпадает) — игнорируем
    has_brief_in_history = False
    if brief_idx is not None:
        brief_content = history[brief_idx].content.lower()
        # Слова запроса (без коротких служебных)
        meaningful = [w for w in query.lower().split() if len(w) > 2]
        brief_is_relevant = any(w in brief_content for w in meaningful) if meaningful else True
        if not brief_is_relevant:
            logger.info("📋 [BRIEF] Previous brief is from different session, ignoring (query=%r)", query[:50])
        has_brief_in_history = brief_is_relevant

    # P5: Извлекаем ответы пользователя, написанные ПОСЛЕ релевантного брифа
    briefing_answers = state.get("briefing_answers", "")
    if has_brief_in_history and not briefing_answers:
        for m in history[brief_idx + 1:]:
            if isinstance(m, HumanMessage) and not _is_pure_confirmation(m.content):
                briefing_answers = m.content
                logger.info("📋 [BRIEF] Extracted briefing answers from history: %.80s...", briefing_answers)
                break

    # P4: Точная проверка подтверждения (не substring!)
    if briefing_answers or has_brief_in_history or _is_pure_confirmation(query):
        return {
            "briefing_questions": state.get("briefing_questions", []),
            "briefing_answers": briefing_answers,
        }

    prompt = prompts.research_brief_prompt.get_prompt(query=query)
    
    llm = _get_llm(state.get("model"))
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = _clean_llm_output(response.content)
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            res = json.loads(match.group())
            if res.get("needs_briefing") and res.get("questions"):
                questions = res["questions"]
                status = f"📋 **Бриф исследования.** Чтобы результат был идеальным, ответьте, пожалуйста, на несколько вопросов:\n\n"
                for i, q in enumerate(questions, 1):
                    status += f"  {i}. {q}\n"
                status += "\nВы можете ответить на всё сразу или просто написать **«Поехали»**, если детали не важны."
                
                new_state = {
                    "briefing_questions": questions,
                    "status": status,
                    "needs_confirmation": True,
                    "messages": [AIMessage(content=status)]
                }
                await update_planner_md({**state, **new_state}, "brief")
                return new_state
    except Exception as e:
        logger.error("Error in brief_node: %s", e)

    return {"briefing_questions": []}


async def planner_node(state: ResearchState) -> dict:
    """
    Разбивает пользовательский запрос на подвопросы для исследования.
    """
    query = state["query"]
    t = query.lower().strip()
    
    # ── Проверка на приветствия и пустые запросы (Chit-chat) ────────
    greetings = ["привет", "хай", "здравствуй", "ку", "hello", "hi", "добрый день", "доброе утро"]
    if t in greetings or len(t) < 3:
        status = "👋 Привет! Я — твой ассистент-исследователь. Готов провести глубокий анализ любой темы.\n\nЧто именно мы будем сегодня исследовать? Введи запрос, и я составлю план."
        return {
            "status": status,
            "sub_questions": [],
            "current_idx": 0,
            "needs_confirmation": True, # Останавливаемся здесь
            "plan_confirmed": False,
            "messages": [AIMessage(content=status)]
        }

    max_q = state.get("max_questions", settings.RESEARCH_MAX_QUESTIONS)
    memory = state.get("memory_context", "")

    from datetime import datetime
    current_year = datetime.now().year

    # Собираем контекст из брифинга ТОЛЬКО если он релевантен текущему запросу
    briefing_info = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage) and "Бриф исследования" in m.content:
            meaningful = [w for w in query.lower().split() if len(w) > 2]
            if meaningful and any(w in m.content.lower() for w in meaningful):
                briefing_info = f"Контекст брифинга: {m.content}\n"
            break

    # P6: Передаём briefing_answers в плановщик (теперь влияют на план)
    prompt = prompts.research_planner_prompt.get_prompt(
        query=query,
        max_questions=max_q,
        current_year=current_year,
        briefing_info=briefing_info,
        memory=memory,
        briefing_answers=state.get("briefing_answers", ""),
    )

    llm = _get_llm(state.get("model"))
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = _clean_llm_output(response.content)

        # Извлекаем JSON из ответа
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            sub_questions = json.loads(match.group())
            # Убедимся что это список строк
            sub_questions = [str(q) for q in sub_questions if q]
        else:
            # Fallback: разбиваем по строкам
            sub_questions = [
                q.strip("- •").strip()
                for q in content.split("\n")
                if q.strip() and not q.strip().startswith(("{", "[", "]"))
            ]

        # Ограничиваем количество
        sub_questions = sub_questions[:max_q]
        logger.info("📅 [PLANNER] Generated %d sub-questions", len(sub_questions))

    except Exception as e:
        logger.error("❌ [PLANNER] Error: %s", e)
        sub_questions = [query]

    # ── Проверка: не подтверждение ли это уже имеющегося плана? ────
    history = state.get("messages", [])
    # P7: Точная проверка — только короткие (≤ 3 слова) чисто-подтверждающие фразы
    is_confirm = _is_pure_confirmation(query)
    
    # Пытаемся найти предыдущий план в истории если это подтверждение
    if is_confirm and len(history) > 1:
        prev_assistant_msg = ""
        for m in reversed(history):
            if isinstance(m, AIMessage):
                prev_assistant_msg = m.content
                break
        
        # Если в предыдущем сообщении был список (проверяем наличие ключевой фразы или просто списка 1.)
        plan_markers = ["Я подготовил план исследования", "План исследования:", "подготовил план"]
        has_plan_marker = any(marker in prev_assistant_msg for marker in plan_markers)
        
        if has_plan_marker:
            logger.info("✅ [PLANNER] Plan confirmed via history")
            # Пытаемся извлечь вопросы из текста если они потерялись (но они должны быть в state если мы используем checkpointing, 
            # но тут мы перезапускаем граф, так что восстанавливаем из текста)
            lines = prev_assistant_msg.split("\n")
            extracted_qs = []
            for line in lines:
                m = re.match(r"^\s*\d+\.\s*(.*)", line)
                if m:
                    extracted_qs.append(m.group(1).strip())
            
            # P10: используем извлечённые вопросы или текущие из стейта (если парсинг не удался)
            qs_to_use = extracted_qs or state.get("sub_questions", [])
            if qs_to_use:
                # Восстанавливаем оригинальный запрос — иначе writer напишет "отчёт по запросу 'старт'"
                original_query = query
                for hm in reversed(history):
                    if isinstance(hm, HumanMessage) and not _is_pure_confirmation(hm.content):
                        original_query = hm.content
                        break
                logger.info("✅ [PLANNER] Restored original query: %.60s", original_query)
                return {
                    "query": original_query,
                    "sub_questions": qs_to_use,
                    "plan_confirmed": True,
                    "needs_confirmation": False,
                    "status": "🚀 План подтвержден, начинаю поиск...",
                }

    # Если это новый запрос — выдаем план и просим подтверждения
    status = f"📊 **Я подготовил план исследования:**\n"
    for i, q in enumerate(sub_questions, 1):
        status += f"  {i}. {q}\n"
    status += "\nЕсли всё верно, напиши **«Поехали»**. Или укажи, что нужно изменить."

    new_state = {
        "sub_questions": sub_questions,
        "current_idx": 0,
        "findings": [],
        "status": status,
        "iteration": 0,
        "plan_confirmed": False,
        "needs_confirmation": True,
        "messages": [AIMessage(content=status)]
    }
    await update_planner_md(new_state, "planner")
    return new_state


async def researcher_node(state: ResearchState) -> dict:
    """
    Исследует текущий подвопрос: поиск → парсинг → извлечение фактов.
    Использует structured-функции для валидации URL.
    """
    from datetime import datetime

    idx = state["current_idx"]
    sub_questions = state["sub_questions"]
    max_urls = state.get("max_urls", settings.RESEARCH_MAX_URLS)
    current_year = datetime.now().year

    if idx >= len(sub_questions):
        return {"status": "✅ Все подвопросы исследованы"}

    question = sub_questions[idx]
    total = len(sub_questions)
    done = idx + 1

    # Прогресс-бар ASCII
    bar_filled = int((done / max(total, 1)) * 12)
    bar = "█" * bar_filled + "░" * (12 - bar_filled)
    pct = int((done / max(total, 1)) * 100)

    status_parts = [f"\n`[{bar}] {pct}%` — 🔍 **[{done}/{total}]** «{question}»"]

    # 1. Поиск через structured API (Pydantic)
    search_response = await web_tools.smart_search_structured(
        question, max_results=max_urls + 2
    )
    found_count = len(search_response.results)
    status_parts.append(f"  ⤷ 📡 Найдено результатов: **{found_count}**")

    # 2. Берём URL и жестко фильтруем от мусора
    raw_urls = search_response.get_urls()
    
    _JUNK_DOMAINS = {
        "facebook.com", "vk.com", "youtube.com", "support.google.com", 
        "linkedin.com", "chatgpt.com", "giga.chat", "instagram.com",
        "twitter.com", "x.com", "tiktok.com", "reddit.com", "quora.com",
        "dzen.ru", "vk.ru", "rutube.ru", "yandex.ru/q", "otvet.mail.ru"
    }

    filtered_urls = []
    for url in raw_urls:
        domain = web_tools._extract_domain(url).lower()
        # Исключаем соцсети, чаты и саппорт
        if any(junk in domain for junk in _JUNK_DOMAINS):
            continue
        filtered_urls.append(url)
        
    urls = filtered_urls[:max_urls]

    # 3. Парсим страницы через structured API
    scraped_pages = []
    ok_count = 0
    for url in urls:
        domain = web_tools._extract_domain(url)
        status_parts.append(f"  ⤷ 📄 `{domain}`")
        scrape_result = await web_tools.smart_scrape_structured(url, max_chars=4000)
        if scrape_result.success:
            scraped_pages.append(scrape_result)
            ok_count += 1
        else:
            status_parts.append(f"    ⚠️ недоступна")

    # 4. Извлекаем факты через LLM
    context = f"Результаты поиска:\n{search_response.to_context_string()}\n\n"
    if scraped_pages:
        context += "Содержимое найденных страниц:\n"
        for page in scraped_pages:
            context += page.to_context_string() + "\n\n"

    extract_prompt = f"""Извлеки ключевые факты и информацию по вопросу:
«{question}»

Текущий год: {current_year}. Используй ТОЛЬКО актуальные данные.

Контекст:
{context}

Правила:
- ВЫДЕЛИ 5-10 конкретных, содержательных фактов
- ОБЯЗАТЕЛЬНО укажи источники для КАЖДОГО факта: заголовок и полная прямая ссылка (URL)
- Ссылки оформляй как [Название](URL)
- Если информация устарела — помечай это
- Если информация не найдена или страницы не открылись — честно об этом напиши
- Отвечай на русском языке
- Будь максимально подробным в деталях
- НЕ используй китайские иероглифы"""

    llm = _get_llm(state.get("model"))
    try:
        response = await llm.ainvoke([HumanMessage(content=extract_prompt)])
        facts = _clean_llm_output(response.content)
    except Exception as e:
        facts = f"⚠️ Ошибка извлечения фактов: {str(e)[:200]}"

    status_parts.append(f"  → ✅ Обработано: {ok_count}/{len(urls)} страниц | {len(facts)} симв. фактов")

    findings = list(state.get("findings", []))
    findings.append(f"### Подвопрос: {question}\n\n{facts}")

    logger.info("🔎 [RESEARCHER] Step %d/%d done | %d URLs scraped", 
                idx + 1, len(sub_questions), len(scraped_pages))

    res = {
        "findings": findings,
        "current_idx": idx + 1,
        "iteration": state.get("iteration", 0) + 1,
        "status": "\n".join(status_parts),
    }
    await update_planner_md({**state, **res}, "researcher")
    return res


async def reflector_node(state: ResearchState) -> dict:
    """
    Нода осмысления: проверяет, достаточно ли информации для финального отчёта.
    Если нет — может добавить новые подвопросы.
    """
    logger.info("🧠 [REFLECTOR] Evaluating research progress...")
    query = state["query"]
    findings = "\n\n".join(state.get("findings", []))
    iteration = state.get("iteration", 0)
    reflector_cycles = state.get("retry_count", 0)  # reuse retry_count as reflector cycle counter

    # Лимит: не более 1 доп. цикла рефлектии, или > 7 итераций
    if reflector_cycles >= 1 or iteration >= 7:
        logger.info("⏹️ [REFLECTOR] Forcing writer after %d cycles / %d iterations", reflector_cycles, iteration)
        return {"status": "⏳ Перехожу к синтезу отчёта...", "retry_count": reflector_cycles + 1}

    prompt = prompts.research_reflector_prompt.get_prompt(query=query, findings=findings)

    llm = _get_llm(state.get("model"))
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = _clean_llm_output(response.content)
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            result = json.loads(match.group())
            is_sufficient = result.get("is_sufficient", True)
            new_questions = result.get("new_questions", [])
            reasoning = result.get("reasoning", "")
        else:
            is_sufficient = True
            new_questions = []
            reasoning = "Не удалось распарсить оценку, завершаем."
    except Exception as e:
        logger.error("Error in reflector: %s", e)
        is_sufficient = True
        new_questions = []
        reasoning = f"Ошибка: {e}"

    if not is_sufficient and new_questions:
        current_sub = list(state.get("sub_questions", []))
        added_count = 0
        for nq in new_questions:
            if nq not in current_sub:
                current_sub.append(nq)
                added_count += 1

        status = f"\n> 🧠 **Анализ:** {reasoning}\n> 🔎 Добавляю {added_count} доп. подвопроса..."
        res = {
            "sub_questions": current_sub,
            "status": status,
            "last_reflection": reasoning,
            "retry_count": reflector_cycles + 1,
        }
        await update_planner_md({**state, **res}, "reflector")
        return res

    res = {"status": f"> ✅ Информации достаточно: {reasoning}"}
    await update_planner_md({**state, **res}, "reflector")
    return res


def reviewer_route(state: ResearchState) -> str:
    """
    Определяет следующий шаг: ещё исследовать или писать отчёт.
    """
    idx = state.get("current_idx", 0)
    sub_questions = state.get("sub_questions", [])

    if idx < len(sub_questions):
        return "researcher"
    return "writer"


async def writer_node(state: ResearchState) -> dict:
    """Финальная нода: пишет итоговый отчет."""
    logger.info("✍️ [WRITER] Synthesizing final report...")
    from datetime import datetime
    query = state["query"]
    findings = state.get("findings", [])

    if not findings:
        return {
            "report": "К сожалению, не удалось найти информацию по вашему запросу.",
            "status": "❌ Нет данных для отчёта",
        }

    all_findings = "\n\n".join(findings)

    synthesis_prompt = prompts.research_writer_prompt.get_prompt(
        query=query, 
        findings=all_findings, 
        current_year=datetime.now().year
    )

    llm = _get_llm(state.get("model"))
    try:
        response = await llm.ainvoke([HumanMessage(content=synthesis_prompt)])
        report = _clean_llm_output(response.content)
    except Exception as e:
        report = f"Ошибка синтеза отчёта: {str(e)[:300]}"

    logger.info("Writer: report generated, %d chars", len(report))

    return {
        "report": report,
        "status": "\n📊 Отчёт готов!",
        "messages": [AIMessage(content=report)],
    }


# ═══════════════════════════════════════════════════════════════════
# Сборка графа
# ═══════════════════════════════════════════════════════════════════


def _clarify_route(state: ResearchState):
    """Решает: нужно ли уточнение или идем в бриф."""
    if state.get("is_ambiguous"):
        return "end"
    return "brief"


def _brief_route(state: ResearchState):
    """Решает: останавливаться для брифинга или идти в план."""
    # Если вопросы сгенерированы, но ответов еще нет — завершаем стрим
    if state.get("briefing_questions") and not state.get("briefing_answers") and state.get("needs_confirmation"):
        return "end"
    return "planner"


def _confirm_route(state: ResearchState):
    """Решает: остановиться для подтверждения или идти в поиск."""
    if state.get("needs_confirmation"):
        return "end"
    return "researcher"


def _reviewer_route(state: ResearchState):
    """Решает: продолжать исследование, идти в рефлектор или писать отчёт."""
    idx = state.get("current_idx", 0)
    sub_questions = state.get("sub_questions", [])

    if idx < len(sub_questions):
        return "researcher"
    
    # Если закончили список, идем в reflector для оценки
    return "reflector"

def _final_route(state: ResearchState):
    """Определяет, нужен ли писатель или еще исследования после рефлектора."""
    idx = state.get("current_idx", 0)
    sub_questions = state.get("sub_questions", [])
    
    if idx < len(sub_questions):
        return "researcher"
    return "writer"


def build_research_graph():
    """Собирает LangGraph граф для Deep Research."""
    graph = StateGraph(ResearchState)

    graph.add_node("clarify", clarify_node)
    graph.add_node("brief", brief_node)
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("reflector", reflector_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("clarify")

    # Маршрут после уточнения
    graph.add_conditional_edges(
        "clarify",
        _clarify_route,
        {
            "end": END,
            "brief": "brief"
        }
    )

    # Маршрут после брифа
    graph.add_conditional_edges(
        "brief",
        _brief_route,
        {
            "end": END,
            "planner": "planner"
        }
    )

    # Маршрут после планировщика
    graph.add_conditional_edges(
        "planner",
        _confirm_route,
        {
            "end": END,
            "researcher": "researcher"
        }
    )

    # Цикл исследования: после каждого Researcher пробуем идти дальше или в Reflector
    graph.add_conditional_edges(
        "researcher",
        _reviewer_route,
        {
            "researcher": "researcher",
            "reflector": "reflector",
        },
    )
    
    # После рефлектора либо снова в поиск (если появились новые вопросы), либо в отчет
    graph.add_conditional_edges(
        "reflector",
        _final_route,
        {
            "researcher": "researcher",
            "writer": "writer",
        }
    )

    graph.add_edge("writer", END)

    return graph.compile()


# Граф создаётся при импорте (не зависит от MCP — tools вызываются runtime)
research_graph = build_research_graph()


# ═══════════════════════════════════════════════════════════════════
# SSE Streaming wrapper
# ═══════════════════════════════════════════════════════════════════


def _make_sse_chunk(content: str, model: str = "deep-research") -> str:
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


def _make_sse_finish(model: str = "deep-research") -> str:
    """Создаёт финальный SSE chunk."""
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


async def deep_research_stream(
    query: str,
    user_id: str,
    messages: list[dict] | None = None,
    model: str = "",
    memory_context: str = "",
    max_questions: int | None = None,
    max_urls: int | None = None,
) -> AsyncGenerator[str, None]:
    """
    Запускает Deep Research и стримит прогресс через SSE.

    Используется в routes.py:
        return StreamingResponse(
            deep_research_stream(query, user_id, ...),
            media_type="text/event-stream",
        )
    """
    _settings = get_settings()
    sse_model = model or "deep-research"

    # Конвертируем историю в объекты LangChain
    history = []
    
    # Всегда добавляем системный промпт в начало для соблюдения правил (Russian, <think> и т.д.)
    sys_prompt = prompts.system_prompt.get_prompt(memory_context=memory_context)
    history.append(SystemMessage(content=sys_prompt))

    if messages:
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                history.append(HumanMessage(content=content))
            elif role == "assistant":
                history.append(AIMessage(content=content))
    else:
        # Если истории нет, добавляем текущий запрос как первое сообщение
        history.append(HumanMessage(content=query))

    initial_state: ResearchState = {
        "messages": history,
        "query": query,
        "sub_questions": [],
        "current_idx": 0,
        "findings": [],
        "report": "",
        "status": "",
        "model": model or _settings.LLM_MODEL,
        "user_id": user_id,
        "memory_context": memory_context,
        "max_questions": max_questions or _settings.RESEARCH_MAX_QUESTIONS,
        "max_urls": max_urls or _settings.RESEARCH_MAX_URLS,
        "iteration": 0,
        "plan_confirmed": False,
        "needs_confirmation": False,
        "last_reflection": "",
        "retry_count": 0,
        "is_ambiguous": False,
        "clarification_options": [],
        "briefing_questions": [],
        "briefing_answers": "",
    }

    # P15: Стабильный thread_id (не случайный uuid) — одинаковый запрос = то же состояние
    _query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
    config = {
        "configurable": {
            "thread_id": f"research-{user_id}-{_query_hash}"
        }
    }

    yield _make_sse_chunk(f"🔬 **Deep Research**: начинаю исследование\n\n", sse_model)

    # Определяем отображаемую тему — если "поехали", берём реальный запрос из истории
    display_query = query
    if _is_pure_confirmation(query):
        for m in reversed(history):
            if isinstance(m, HumanMessage) and not _is_pure_confirmation(m.content):
                display_query = m.content[:120]
                break

    yield _make_sse_chunk(f"📋 Запрос: «{display_query}»\n\n---\n\n", sse_model)

    try:
        async for event in research_graph.astream(initial_state, config):
            for node_name, node_output in event.items():
                if node_name == "__end__":
                    continue

                status = node_output.get("status", "")
                if status:
                    yield _make_sse_chunk(status + "\n\n", sse_model)

                # Writer — стримим отчёт
                if node_name == "writer":
                    report = node_output.get("report", "")
                    if report:
                        yield _make_sse_chunk(
                            "\n---\n\n# 📊 Результаты исследования\n\n", sse_model
                        )
                        # Поабзацный стриминг отчёта параграфами
                        paragraphs = report.split("\n\n")
                        for para in paragraphs:
                            if para.strip():
                                yield _make_sse_chunk(para + "\n\n", sse_model)
                        yield _make_sse_chunk("\n---\n📝 _Лог планирования сохранён на сервере._\n", sse_model)

    except Exception as e:
        logger.error("Deep Research failed: %s", e, exc_info=True)
        yield _make_sse_chunk(f"\n\n❌ Ошибка исследования: {str(e)[:300]}", sse_model)

    yield _make_sse_finish(sse_model)
    yield "data: [DONE]\n\n"
