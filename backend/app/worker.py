"""Worker — мост между RabbitMQ и LangGraph Server.

Слушает прикладную очередь RabbitMQ, на каждую задачу вызывает граф через SDK и
кладёт результат в Redis. Это отдельный процесс/контейнер: ровно так HTTP-ручка
API развязывается с тяжёлой работой графа.

Запуск: python -m app.worker
"""

from __future__ import annotations

import asyncio
import json

import aio_pika

from .config import settings
from .lg_client import aclose_client, run_graph
from .mq import aclose_mq, set_job


async def _handle(message: aio_pika.abc.AbstractIncomingMessage) -> None:
    # message.process() подтвердит (ack) сообщение при успехе и вернёт в очередь
    # при исключении — задача не потеряется.
    async with message.process():
        data = json.loads(message.body)
        job_id = data["job_id"]
        await set_job(job_id, {"status": "running", "answer": None})
        try:
            answer = await run_graph(data["message"])
            await set_job(job_id, {"status": "done", "answer": answer})
        except Exception as exc:  # демо: просто фиксируем ошибку в статусе задачи
            await set_job(job_id, {"status": "error", "answer": str(exc)})


async def main() -> None:
    # Отдельное соединение для consume: канал воркера со своим QoS живёт всё
    # время процесса (его держит сам connect_robust). Общие клиенты для записи
    # результата (LangGraph SDK, Redis) поднимаются лениво внутри _handle.
    connection = await aio_pika.connect_robust(settings.rabbitmq_url)
    try:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=4)  # не более 4 задач в работе одновременно
        queue = await channel.declare_queue(settings.job_queue, durable=True)
        await queue.consume(_handle)
        print(f"[worker] слушаю очередь '{settings.job_queue}'…")
        await asyncio.Future()  # блокируемся навсегда
    finally:
        await aclose_client()
        await aclose_mq()
        await connection.close()


if __name__ == "__main__":
    asyncio.run(main())
