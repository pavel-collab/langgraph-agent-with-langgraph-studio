"""Тонкая обёртка над langgraph-sdk.

Главная идея проекта: backend не импортирует код графа напрямую, а общается с
развёрнутым LangGraph Server по сети через официальный SDK. Сервер сам поднимает
REST API над скомпилированным графом, а мы лишь создаём поток (thread) и
запускаем run.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph_sdk import get_client
from langgraph_sdk.client import LangGraphClient

from .config import settings


@lru_cache(maxsize=1)
def _client() -> LangGraphClient:
    """Один экземпляр клиента на процесс.

    Внутри LangGraphClient живёт httpx.AsyncClient со своим пулом keep-alive
    соединений. Раньше клиент создавался на каждый вызов run_graph — пул не
    переиспользовался, на каждый запрос шёл новый TCP-handshake, а сам клиент
    оставался незакрытым до сборки мусора. Singleton снимает обе проблемы.

    Ленивое создание (через lru_cache при первом обращении из run_graph)
    гарантирует, что httpx-клиент инициализируется уже внутри работающего
    event loop — и backend, и worker живут каждый в одном loop весь свой срок,
    поэтому один общий клиент безопасен.
    """
    return get_client(url=settings.langgraph_url)


async def aclose_client() -> None:
    """Закрыть внутренний httpx.AsyncClient. Зовётся на остановке процесса."""
    if _client.cache_info().currsize:  # клиент вообще создавали?
        await _client().http.client.aclose()
        _client.cache_clear()


async def run_graph(message: str) -> str:
    """Прогнать граф на одном сообщении и вернуть текст финального ответа.

    runs.wait создаёт run и блокируется до его завершения — синхронный вызов
    с точки зрения backend. Внутри же LangGraph Server всё равно проводит run
    через свою Redis-очередь и Postgres-чекпоинты.
    """
    client = _client()

    thread = await client.threads.create()
    result = await client.runs.wait(
        thread["thread_id"],
        settings.graph_id,
        input={"messages": [{"role": "human", "content": message}]},
    )

    messages = result.get("messages", []) if isinstance(result, dict) else []
    if not messages:
        return ""
    last = messages[-1]
    return last.get("content", "") if isinstance(last, dict) else str(last)
