"""Конфигурация backend — всё через переменные окружения (12-factor).

Имена полей читаются из ENV без учёта регистра: langgraph_url ← LANGGRAPH_URL.
Дефолты указывают на имена сервисов из docker-compose, чтобы внутри сети Docker
ничего дополнительно настраивать не пришлось.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Куда backend ходит за графом. Это внутренний REST API LangGraph Server,
    # который мы НЕ публикуем наружу — наружу смотрит только этот backend.
    langgraph_url: str = "http://langgraph-server:8000"
    graph_id: str = "research_assistant"

    # Брокер прикладного уровня: ручка API → очередь → worker.
    rabbitmq_url: str = "amqp://guest:guest@rabbitmq:5672/"
    job_queue: str = "research_jobs"

    # Хранилище статусов/результатов задач. Берём ОТДЕЛЬНУЮ логическую БД (/1),
    # чтобы не пересекаться с Redis, который LangGraph Server использует под
    # свою внутреннюю очередь run-ов (та живёт в БД по умолчанию, /0).
    redis_url: str = "redis://redis:6379/1"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
