"""Субагенты-специалисты, собранные через create_agent.

`create_agent` (LangChain 1.0) — это «фабрика» готового ReAct-агента: ты даёшь
модель, список инструментов и системный промпт, а на выходе получаешь
СКОМПИЛИРОВАННЫЙ граф (Runnable). Внутри он сам гоняет цикл
    модель → (вызвать инструмент?) → инструмент → модель → … → ответ,
поэтому отдельный StateGraph под каждого специалиста писать не нужно.

Каждый такой субагент принимает на вход {"messages": [...]} и возвращает
{"messages": [...]} — тот же интерфейс, что и у узла обычного графа. Благодаря
этому в graph.py мы вставляем субагентов как обычные узлы, склеивая их вручную.

Здесь три специалиста, у каждого СУЖЕННЫЙ набор инструментов:
- search_agent     — ищет статьи (search_arxiv + get_paper_details);
- summarizer_agent — разбирает/конспектирует (get_paper_details);
- calc_agent       — считает и сравнивает метрики (calculator).

ВАЖНО про Runtime[Context]: субагенты больше не создаются как глобалы на этапе
импорта. Вместо этого здесь лежат фабрики make_*_agent(model, temperature,
system_prompt). Узлы графа вызывают их с параметрами из runtime.context, поэтому
системный промпт и модель можно подменять на лету, без перезапуска процесса.
Фабрики обёрнуты в lru_cache: при одинаковых параметрах агент собирается один
раз и переиспользуется (агенты не хранят состояние между вызовами).
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from agent.tools import calculator, get_paper_details, search_arxiv


def make_model(model: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    """Создать клиент модели по переменным окружения (как в статье).

    base_url / api_key берём из .env, чтобы легко переключаться между OpenAI,
    локальной моделью (llama.cpp, vLLM) или корпоративным шлюзом. Имя модели
    можно переопределить аргументом `model` (приходит из Context); если он не
    задан — берётся LLM_NAME из окружения.
    """
    return ChatOpenAI(
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_KEY", "not-needed"),
        model=model or os.getenv("LLM_NAME", "gpt-4o-mini"),
        temperature=temperature,
    )


# --- Системные промпты по умолчанию -----------------------------------------
# Вынесены в константы, чтобы служить дефолтом, когда Context их не переопределяет.

DEFAULT_SEARCH_SYSTEM = (
    "Ты — помощник по поиску научных статей на arXiv. "
    "Сначала вызови search_arxiv с осмысленным англоязычным запросом, "
    "при необходимости уточни детали через get_paper_details. "
    "В ответе перечисли найденные статьи с arXiv id и краткой пользой каждой. "
    "Отвечай на русском."
)

DEFAULT_SUMMARIZE_SYSTEM = (
    "Ты — помощник, который разбирает научные статьи. "
    "Если пользователь дал arXiv id — подтяни аннотацию через get_paper_details. "
    "Сделай структурированный конспект: задача, метод, ключевой результат, ограничения. "
    "Пиши кратко и по делу, на русском."
)

DEFAULT_CALC_SYSTEM = (
    "Ты — помощник по расчётам над числами из статей "
    "(приросты метрик, разницы, проценты). "
    "Любую арифметику считай ТОЛЬКО через инструмент calculator, не в уме. "
    "Покажи выражение и результат, поясни смысл. Отвечай на русском."
)


# --- Фабрики субагентов ------------------------------------------------------
# lru_cache: одинаковые (model, temperature, system_prompt) → тот же агент.
# Аргументы хэшируемы (строки/число/None), поэтому кэш работает корректно.

# lry cache гарантирует, что при одинаковых параметрах мы не будем создавать несколько экземпляров агента при каждом обращении к нему в рамках одной сессии
@lru_cache(maxsize=16)
def make_search_agent(
    model: str | None = None,
    temperature: float = 0.0,
    system_prompt: str = DEFAULT_SEARCH_SYSTEM,
):
    """Специалист 1: поиск статей (search_arxiv + get_paper_details)."""
    return create_agent(
        make_model(model, temperature),
        tools=[search_arxiv, get_paper_details],
        system_prompt=system_prompt,
    )


@lru_cache(maxsize=16)
def make_summarizer_agent(
    model: str | None = None,
    temperature: float = 0.0,
    system_prompt: str = DEFAULT_SUMMARIZE_SYSTEM,
):
    """Специалист 2: разбор и конспект (get_paper_details)."""
    return create_agent(
        make_model(model, temperature),
        tools=[get_paper_details],
        system_prompt=system_prompt,
    )


@lru_cache(maxsize=16)
def make_calc_agent(
    model: str | None = None,
    temperature: float = 0.0,
    system_prompt: str = DEFAULT_CALC_SYSTEM,
):
    """Специалист 3: расчёты и сравнение метрик (calculator)."""
    return create_agent(
        make_model(model, temperature),
        tools=[calculator],
        system_prompt=system_prompt,
    )
