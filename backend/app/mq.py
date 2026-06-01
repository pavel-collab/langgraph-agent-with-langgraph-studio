"""RabbitMQ (публикация задач) + Redis (хранение статусов/результатов).

Это прикладной слой очередей, отдельный от внутренней Redis-очереди LangGraph
Server. Поток такой:

    POST /chat/async → publish_job() → RabbitMQ → worker → run_graph() → set_job()
    GET  /jobs/{id}  → get_job() (читает результат из Redis)
"""

from __future__ import annotations

import asyncio
import json

import aio_pika
import redis.asyncio as aioredis

from .config import settings


# --- RabbitMQ ---------------------------------------------------------------
# Раньше publish_job на каждый вызов открывал новое AMQP-соединение и канал.
# Держим одно robust-соединение и один канал на процесс: connect_robust сам
# переподключается при обрыве, а очередь объявляем один раз при инициализации.
_amqp_connection: aio_pika.abc.AbstractRobustConnection | None = None
_amqp_channel: aio_pika.abc.AbstractChannel | None = None
_amqp_lock = asyncio.Lock()


async def _channel() -> aio_pika.abc.AbstractChannel:
    """Ленивая инициализация общего канала (потокобезопасно в рамках loop)."""
    global _amqp_connection, _amqp_channel
    if _amqp_channel is not None and not _amqp_channel.is_closed:
        return _amqp_channel
    async with _amqp_lock:
        if _amqp_channel is not None and not _amqp_channel.is_closed:
            return _amqp_channel
        _amqp_connection = await aio_pika.connect_robust(settings.rabbitmq_url)
        _amqp_channel = await _amqp_connection.channel()
        await _amqp_channel.declare_queue(settings.job_queue, durable=True)
    return _amqp_channel


async def publish_job(job_id: str, message: str) -> None:
    """Положить задачу в durable-очередь. Сообщение persistent — переживёт рестарт."""
    channel = await _channel()
    await channel.default_exchange.publish(
        aio_pika.Message(
            body=json.dumps({"job_id": job_id, "message": message}).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key=settings.job_queue,
    )


# --- Redis (хранилище статусов задач) ---------------------------------------
# from_url создаёт клиент с собственным пулом соединений — держим один на
# процесс вместо нового клиента (и нового пула) на каждый set/get.
_redis_client: aioredis.Redis | None = None


def _redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis_client


async def set_job(job_id: str, payload: dict) -> None:
    """Сохранить статус/результат задачи на час (демо — без вечного хранения)."""
    await _redis().set(f"job:{job_id}", json.dumps(payload), ex=3600)


async def get_job(job_id: str) -> dict | None:
    raw = await _redis().get(f"job:{job_id}")
    return json.loads(raw) if raw else None


# --- Закрытие ресурсов на остановке процесса --------------------------------
async def aclose_mq() -> None:
    """Закрыть общие подключения. Зовётся из lifespan backend-а и из worker-а."""
    global _amqp_connection, _amqp_channel, _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
    if _amqp_connection is not None:
        await _amqp_connection.close()
        _amqp_connection = None
        _amqp_channel = None
