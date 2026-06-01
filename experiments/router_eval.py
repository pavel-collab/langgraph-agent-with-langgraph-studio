"""Эксперимент №1: точность узла-роутера на датасете router_testset.

Что здесь происходит (по шагам):

1. Заливаем (или переиспользуем) датасет `router-testset` в LangSmith из
   локального файла datasets/router_testset.jsonl. Каждый пример — это
   {"inputs": {"messages": [...]}, "outputs": {"route": "search"|...}}.

2. Описываем `target` — что именно мы прогоняем на каждом примере. Нас интересует
   ровно одно решение: куда роутер направит запрос. Поэтому target повторяет
   логику router_node из graph.py (structured output → одно из 4 значений),
   но НЕ запускает субагентов с инструментами — так эксперимент дёшев и быстр,
   и измеряет именно качество классификации, а не работу всего графа.

3. Описываем эвалуаторы — функции, которые сравнивают ответ target с эталоном:
   - correct_route   — точное совпадение метки (1/0), главная метрика accuracy;
   - route_label     — кладёт предсказанную метку в фидбэк, чтобы в UI было видно,
                       КУДА именно ушёл каждый запрос (удобно строить confusion).

4. Запускаем aevaluate(...). LangSmith создаёт ЭКСПЕРИМЕНT над датасетом — он
   появляется в LangSmith UI (вкладка Datasets & Experiments) и в LangGraph
   Studio. Там его можно открыть, посмотреть пример-за-примером, сравнить с
   будущими прогонами и перезапустить из интерфейса.

Запуск:
    .venv/bin/python experiments/router_eval.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import Client
from langsmith.evaluation import aevaluate

# .env лежит в корне проекта (на уровень выше этой папки).
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Импорт после load_dotenv: make_model читает LLM_* из окружения.
from agent.graph import DEFAULT_ROUTER_SYSTEM, RouterDecision  # noqa: E402
from agent.subagents import make_model  # noqa: E402

DATASET_NAME = "router-testset"
DATASET_FILE = Path(__file__).resolve().parents[1] / "datasets" / "router_testset.jsonl"
VALID_ROUTES = {"search", "summarize", "calc", "chat"}


# --- Шаг 1: датасет в LangSmith --------------------------------------------
def ensure_dataset(client: Client) -> str:
    """Создать датасет из jsonl, если его ещё нет. Вернуть имя датасета.

    Идемпотентно: если датасет с таким именем уже есть — НЕ заливаем повторно,
    чтобы не плодить дубликаты примеров. Для перезаливки удали датасет в UI.
    """
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Датасет '{DATASET_NAME}' уже существует — переиспользуем.")
        return DATASET_NAME

    rows = [json.loads(line) for line in DATASET_FILE.read_text().splitlines() if line.strip()]
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Тест-сет для узла-роутера research-ассистента: вопрос → ветка.",
    )
    client.create_examples(
        inputs=[r["inputs"] for r in rows],
        outputs=[r["outputs"] for r in rows],
        dataset_id=dataset.id,
    )
    print(f"Создан датасет '{DATASET_NAME}' ({len(rows)} примеров).")
    return DATASET_NAME


# --- Шаг 2: target (повторяет логику router_node) ---------------------------
async def route_target(inputs: dict) -> dict:
    """Прогнать роутер на одном примере. Возвращает {'route': <ветка>}."""
    last = inputs["messages"][-1]
    content = last["content"] if isinstance(last, dict) else last.content

    # temperature=0 — классификация должна быть стабильной (как в router_node).
    router_llm = make_model(None, 0.0).with_structured_output(RouterDecision)
    decision: RouterDecision = await router_llm.ainvoke(
        [SystemMessage(content=DEFAULT_ROUTER_SYSTEM), HumanMessage(content=content)]
    )
    route = decision.destination if decision.destination in VALID_ROUTES else "chat"
    return {"route": route}


# --- Шаг 3: эвалуаторы -------------------------------------------------------
def correct_route(outputs: dict, reference_outputs: dict) -> dict:
    """Главная метрика: предсказанная ветка совпала с эталонной (1) или нет (0)."""
    score = int(outputs["route"] == reference_outputs["route"])
    return {"key": "correct_route", "score": score}


def route_label(outputs: dict) -> dict:
    """Вспомогательный фидбэк: какую ветку выбрал роутер (для confusion в UI)."""
    return {"key": "predicted_route", "value": outputs["route"]}


# --- Шаг 4: запуск эксперимента ---------------------------------------------
async def main() -> None:
    client = Client()
    dataset_name = ensure_dataset(client)

    print("Запускаю эксперимент…")
    results = await aevaluate(
        route_target,
        data=dataset_name,
        evaluators=[correct_route, route_label],
        experiment_prefix="router-accuracy",
        max_concurrency=4,
        client=client,
    )

    # Короткая сводка в консоль (полные результаты — в UI).
    print(f"\nЭксперимент: {results.experiment_name}")
    print("Готово. Открой его в LangSmith UI / LangGraph Studio во вкладке Experiments.")


if __name__ == "__main__":
    asyncio.run(main())
