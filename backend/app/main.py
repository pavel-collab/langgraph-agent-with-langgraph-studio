"""FastAPI backend — единственная точка, открытая пользователю.

Намеренно простой (демо). Показывает два способа обратиться к графу:

- POST /chat        — синхронно: backend сам ждёт ответ графа через SDK;
- POST /chat/async  — асинхронно: задача уходит в RabbitMQ, ответ — сразу job_id,
                      результат потом забирается через GET /jobs/{id}.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .lg_client import aclose_client, run_graph
from .mq import aclose_mq, get_job, publish_job, set_job


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Долгоживущие клиенты (LangGraph SDK, RabbitMQ, Redis) создаются лениво
    # при первом запросе и переиспользуются. Здесь лишь аккуратно закрываем их
    # на остановке, чтобы не оставлять висящих соединений.
    yield
    await aclose_client()
    await aclose_mq()


app = FastAPI(title="Research Assistant Backend", version="0.1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Синхронный путь: дождаться графа и сразу вернуть ответ."""
    answer = await run_graph(req.message)
    return ChatResponse(answer=answer)


@app.post("/chat/async")
async def chat_async(req: ChatRequest) -> dict:
    """Асинхронный путь: поставить задачу в очередь и вернуть её id."""
    job_id = str(uuid.uuid4())
    await set_job(job_id, {"status": "queued", "answer": None})
    await publish_job(job_id, req.message)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str) -> dict:
    """Узнать статус/результат ранее поставленной задачи."""
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job_id": job_id, **job}
