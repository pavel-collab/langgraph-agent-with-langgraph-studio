"""Вручную собранный граф research-ассистента.

Идея (по мотивам статьи Selectel про LangGraph Server): запрос пользователя
сначала попадает в узел `router`, который на базе ChatOpenAI решает, в какую
ветку его направить. Ветки — это не «голые» узлы, а самостоятельные субагенты
из subagents.py, созданные через create_agent. Мы вставляем их в граф как
обычные узлы и склеиваем условным ребром.

    START → router ──┬──→ search    ─→ END
                     ├──→ summarize ─→ END
                     ├──→ calc      ─→ END
                     └──→ chat      ─→ END

Так получается «граф из агентов»: верхний уровень — ручная маршрутизация,
нижний — готовые ReAct-циклы внутри каждого субагента.

Runtime[Context]: граф скомпилирован с context_schema=Context, поэтому каждый
узел получает второй аргумент `runtime: Runtime[Context]`. Системные промпты,
имя модели и температуры берутся из `runtime.context` (с откатом на дефолты).
Это позволяет подменять поведение (например, системный промпт) НА ЛЕТУ при
вызове графа, без перезапуска процесса — конфиг едет рядом с запросом, а не
зашит в модуль.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from agent.state import Context, State
from agent.subagents import (
    DEFAULT_CALC_SYSTEM,
    DEFAULT_SEARCH_SYSTEM,
    DEFAULT_SUMMARIZE_SYSTEM,
    make_calc_agent,
    make_model,
    make_search_agent,
    make_summarizer_agent,
)

# --- Узел router ------------------------------------------------------------
# Роутер использует structured output: модель обязана вернуть ровно одно из
# допустимых значений destination, а не свободный текст. Это надёжнее парсинга.


class RouterDecision(BaseModel):
    """Решение роутера: в какую ветку направить запрос."""

    destination: str = Field(
        description=(
            "Куда направить запрос. Одно из: "
            "'search' — найти статьи по теме; "
            "'summarize' — разобрать/сделать конспект конкретной статьи; "
            "'calc' — посчитать или сравнить числовые метрики; "
            "'chat' — всё остальное (общий вопрос, приветствие, уточнение)."
        )
    )


# --- Системные промпты по умолчанию (router и chat) -------------------------
# Дефолты для веток, которыми graph.py управляет напрямую. Любой из них можно
# переопределить через Context, не трогая код.

DEFAULT_ROUTER_SYSTEM = (
    "Ты — маршрутизатор research-ассистента. По последнему сообщению пользователя "
    "определи нужную ветку и верни её в поле destination. "
    "Ничего не выполняй сам — только классифицируй."
)

DEFAULT_CHAT_SYSTEM = (
    "Ты — дружелюбный research-ассистент по научным статьям. "
    "Отвечай кратко и по делу на русском. Если для ответа явно нужен поиск, "
    "разбор статьи или расчёт — подскажи пользователю переформулировать запрос."
)


def _ctx(runtime: Runtime[Context]) -> Context:
    """Безопасно достать context: при вызове без него runtime.context == None."""
    return runtime.context or {}


async def router_node(state: State, runtime: Runtime[Context]) -> dict:
    """Классифицирует последнее сообщение и кладёт решение в state['route']."""
    ctx = _ctx(runtime)

    # temperature=0 по умолчанию — классификация должна быть стабильной.
    llm = make_model(ctx.get("model"), ctx.get("router_temperature", 0.0))
    
    router_llm = llm.with_structured_output(RouterDecision)
    system = ctx.get("router_system") or DEFAULT_ROUTER_SYSTEM

    last_message = state["messages"][-1]
    decision: RouterDecision = await router_llm.ainvoke(
        [SystemMessage(content=system), last_message]
    )
    route = decision.destination if decision.destination in {
        "search", "summarize", "calc", "chat"
    } else "chat"
    return {"route": route}


def pick_route(state: State) -> str:
    """Функция-выбор для условного ребра: просто читает решение роутера."""
    return state["route"]


# --- Узлы-субагенты ---------------------------------------------------------
# Субагент возвращает ВСЮ историю (вход + новые сообщения). Чтобы не плодить
# дубликаты и вернуть в общий граф только то, что субагент добавил, отрезаем
# первые len(входных) сообщений. add_messages затем подмержит их в общую ленту.


async def _run_subagent(agent, state: State) -> dict:
    incoming = state["messages"]
    result = await agent.ainvoke({"messages": incoming})
    new_messages = result["messages"][len(incoming):]
    return {"messages": new_messages}


async def search_node(state: State, runtime: Runtime[Context]) -> dict:
    ctx = _ctx(runtime)
    agent = make_search_agent(
        ctx.get("model"),
        ctx.get("subagent_temperature", 0.0),
        ctx.get("search_system") or DEFAULT_SEARCH_SYSTEM,
    )
    return await _run_subagent(agent, state)


async def summarize_node(state: State, runtime: Runtime[Context]) -> dict:
    ctx = _ctx(runtime)
    agent = make_summarizer_agent(
        ctx.get("model"),
        ctx.get("subagent_temperature", 0.0),
        ctx.get("summarize_system") or DEFAULT_SUMMARIZE_SYSTEM,
    )
    return await _run_subagent(agent, state)


async def calc_node(state: State, runtime: Runtime[Context]) -> dict:
    ctx = _ctx(runtime)
    agent = make_calc_agent(
        ctx.get("model"),
        ctx.get("subagent_temperature", 0.0),
        ctx.get("calc_system") or DEFAULT_CALC_SYSTEM,
    )
    return await _run_subagent(agent, state)


async def chat_node(state: State, runtime: Runtime[Context]) -> dict:
    ctx = _ctx(runtime)
    # temperature=0.7 по умолчанию — для чата нужна «живость».
    chat_llm = make_model(ctx.get("model"), ctx.get("chat_temperature", 0.7))
    system = ctx.get("chat_system") or DEFAULT_CHAT_SYSTEM
    response: AIMessage = await chat_llm.ainvoke(
        [SystemMessage(content=system), *state["messages"]]
    )
    return {"messages": [response]}


# --- Сборка графа -----------------------------------------------------------
# context_schema=Context включает Runtime[Context] для всех узлов.
_builder = StateGraph(State, context_schema=Context)

_builder.add_node("router", router_node)
_builder.add_node("search", search_node)
_builder.add_node("summarize", summarize_node)
_builder.add_node("calc", calc_node)
_builder.add_node("chat", chat_node)

_builder.add_edge(START, "router")
_builder.add_conditional_edges(
    "router",
    pick_route,
    {
        "search": "search",
        "summarize": "summarize",
        "calc": "calc",
        "chat": "chat",
    },
)
_builder.add_edge("search", END)
_builder.add_edge("summarize", END)
_builder.add_edge("calc", END)
_builder.add_edge("chat", END)

# Имя графа видно в LangGraph Studio и LangSmith.
graph = _builder.compile(name="Research Assistant")
