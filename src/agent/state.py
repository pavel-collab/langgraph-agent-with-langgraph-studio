"""Состояние графа.

В LangGraph «состояние» — это словарь, который путешествует между узлами.
Каждый узел получает его на вход и возвращает частичное обновление (patch),
которое LangGraph применяет к общему состоянию.

Поле `messages` помечено редьюсером `add_messages`: это значит, что когда узел
возвращает {"messages": [...]}, новые сообщения не затирают историю, а
аккуратно добавляются к ней (с дедупликацией по id). Так история диалога
накапливается по мере прохождения по графу.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import Field

# Куда роутер может направить запрос. Эти же строки служат именами узлов в графе.
Route = Literal["search", "summarize", "calc", "chat"]


class State(TypedDict):
    """Состояние, общее для всего графа research-ассистента."""

    # История сообщений диалога. add_messages — редьюсер: добавляет, а не заменяет.
    messages: Annotated[list[AnyMessage], add_messages]

    # Решение роутера: имя ветки, в которую уйдёт запрос на этом шаге.
    # total=False тут не используем намеренно — поле всегда выставляет узел router.
    route: Route


class Context(TypedDict, total=False):
    """Конфигурация графа, передаваемая на этапе ВЫЗОВА (а не в State).

    State — это изменяемая «память» (она копится через узлы), а Context —
    неизменяемые на время прогона параметры: системные промпты, имя модели,
    температуры. Узлы получают его как `runtime.context` (см. graph.py).

    Зачем отдельно от State: один и тот же скомпилированный граф можно вызывать
    с РАЗНЫМИ системными промптами/моделью без перезапуска процесса —
    достаточно передать context при invoke или при создании ассистента в
    LangGraph Server. Все поля опциональны (total=False): если поле не задано,
    узел берёт встроенный дефолт.

    Пример вызова:
        graph.ainvoke(
            {"messages": [...]},
            context={"chat_system": "Ты — строгий рецензент.", "model": "gpt-4o"},
        )
    """

    # Имя модели; None/нет ключа → берётся из переменной окружения LLM_NAME.
    # Метка __template_metadata__ kind=llm подсказывает Studio показать селектор
    # модели для этого поля.
    #
    # ВАЖНО: метаданные обязательно через pydantic.Field(json_schema_extra=...).
    # Схему контекста из TypedDict строит Pydantic, и плоский dict внутри
    # Annotated (Annotated[str, {...}]) он игнорирует — ключи не попадут в
    # JSON-схему, и Studio их не увидит. Только json_schema_extra доезжает.
    model: Annotated[
        str, Field(json_schema_extra={"__template_metadata__": {"kind": "llm"}})
    ]

    # Температуры по веткам (у роутера нужна стабильность, у чата — живость).
    router_temperature: float
    chat_temperature: float
    subagent_temperature: float

    # Системные промпты по веткам. Это и есть «горячая» подмена поведения.
    #
    # Метаданные включают вкладку Prompts в LangGraph Studio:
    #   - langgraph_type="prompt" — поле редактируется как промпт в UI;
    #   - langgraph_nodes=[...]   — какие узлы графа используют этот промпт
    #                               (Studio показывает связь поле ↔ узлы).
    # Имена узлов должны совпадать с add_node(...) в graph.py.
    router_system: Annotated[
        str,
        Field(
            json_schema_extra={"langgraph_type": "prompt", "langgraph_nodes": ["router"]}
        ),
    ]
    chat_system: Annotated[
        str,
        Field(
            json_schema_extra={"langgraph_type": "prompt", "langgraph_nodes": ["chat"]}
        ),
    ]
    search_system: Annotated[
        str,
        Field(
            json_schema_extra={"langgraph_type": "prompt", "langgraph_nodes": ["search"]}
        ),
    ]
    summarize_system: Annotated[
        str,
        Field(
            json_schema_extra={
                "langgraph_type": "prompt",
                "langgraph_nodes": ["summarize"],
            }
        ),
    ]
    calc_system: Annotated[
        str,
        Field(
            json_schema_extra={"langgraph_type": "prompt", "langgraph_nodes": ["calc"]}
        ),
    ]
