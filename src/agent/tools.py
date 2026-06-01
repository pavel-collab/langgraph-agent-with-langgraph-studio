"""Инструменты агента.

Инструмент (tool) в LangChain — это обычная Python-функция, обёрнутая
декоратором @tool. Декоратор берёт сигнатуру и docstring функции и
превращает их в JSON-схему, которую модель «видит» и по которой решает,
какой инструмент и с какими аргументами вызвать.

Здесь три инструмента под предметную область «работа с научными статьями»:
- search_arxiv      — поиск статей на arXiv по ключевым словам;
- get_paper_details — получить заголовок/авторов/аннотацию по arXiv id;
- calculator        — безопасный калькулятор для сравнения метрик из статей.

Важно: docstring — это не украшение, а часть промпта для модели.
От его ясности напрямую зависит, правильно ли агент выберет инструмент.

ПОЧЕМУ RSS/Atom, а не пакет `arxiv`:
Пакет `arxiv` оборачивает один и тот же публичный эндпоинт arXiv API, но при
частых вызовах из агента arXiv начинает троттлить наш IP (HTTP 429). Поэтому,
по образцу проекта ArxivNewsPipeline, мы ходим в arXiv API напрямую и парсим
Atom-ответ через `feedparser`. Это убирает лишний слой клиента, даёт контроль
над User-Agent (arXiv просит его указывать) и легче переживает повторные запросы.
"""

from __future__ import annotations

import ast
import operator
import time
from urllib.parse import quote_plus

import feedparser
from langchain_core.tools import tool

# Базовый эндпоинт arXiv API. Отдаёт Atom-фид, который понимает feedparser.
_ARXIV_API_URL = "http://export.arxiv.org/api/query"
# arXiv просит указывать осмысленный User-Agent — так запросы реже попадают
# под троттлинг, чем «безымянный» python-urllib по умолчанию.
_USER_AGENT = "langgraph-research-assistant/0.1 (+https://arxiv.org)"


def _fetch_feed(query_string: str, num_retries: int = 3):
    """Сходить в arXiv API и вернуть распарсенный feedparser-фид.

    query_string — это уже собранная строка query-параметров (без ведущего «?»).
    feedparser сам делает HTTP-запрос и разбирает Atom. На временный троттлинг
    (HTTP 429) делаем несколько повторов с нарастающей паузой.
    """
    url = f"{_ARXIV_API_URL}?{query_string}"
    feed = None
    for attempt in range(num_retries):
        feed = feedparser.parse(url, agent=_USER_AGENT)
        # status есть только при сетевом запросе; 429 = arXiv троттлит наш IP.
        if getattr(feed, "status", 200) == 429:
            time.sleep(3.0 * (attempt + 1))
            continue
        return feed
    return feed


def _parse_entry(entry) -> dict:
    """Привести запись Atom-фида к единому словарю с полями статьи."""
    # entry.id вида "http://arxiv.org/abs/2310.06825v1" → короткий id "2310.06825v1".
    entry_id = entry.get("id", "")
    short_id = entry_id.rsplit("/abs/", 1)[-1] if "/abs/" in entry_id else entry_id

    authors = [a.get("name", "") for a in entry.get("authors", []) if a.get("name")]

    pub_date = ""
    if getattr(entry, "published_parsed", None):
        pub_date = time.strftime("%Y-%m-%d", entry.published_parsed)

    return {
        "id": short_id,
        "title": entry.get("title", "").replace("\n", " ").strip(),
        "authors": authors,
        "date": pub_date,
        "summary": entry.get("summary", "").replace("\n", " ").strip(),
        "link": entry.get("link", entry_id),
    }


@tool
def search_arxiv(query: str, max_results: int = 5) -> str:
    """Найти научные статьи на arXiv по ключевым словам.

    Используй для запросов вида «найди статьи про…», «что есть по теме…».
    Аргументы:
        query: поисковый запрос на английском (напр. "graph retrieval augmented generation").
        max_results: сколько статей вернуть (по умолчанию 5).
    Возвращает список статей: arXiv id, заголовок, авторы, дата и ссылка.
    """
    query_string = (
        f"search_query=all:{quote_plus(query)}"
        f"&start=0&max_results={max_results}"
        f"&sortBy=relevance&sortOrder=descending"
    )
    feed = _fetch_feed(query_string)
    if getattr(feed, "status", 200) == 429:
        return (
            "arXiv временно ограничил частоту запросов (HTTP 429). "
            "Подождите минуту и повторите запрос."
        )
    if not feed.entries:
        return f"По запросу «{query}» ничего не найдено."

    lines = []
    for entry in feed.entries:
        paper = _parse_entry(entry)
        authors = ", ".join(paper["authors"][:3])
        if len(paper["authors"]) > 3:
            authors += " и др."
        lines.append(
            f"- [{paper['id']}] {paper['title']}\n"
            f"  Авторы: {authors}\n"
            f"  Дата: {paper['date']}\n"
            f"  Кратко: {paper['summary'][:200]}...\n"
            f"  Ссылка: {paper['link']}"
        )
    return "\n".join(lines)


@tool
def get_paper_details(arxiv_id: str) -> str:
    """Получить детали одной статьи по её arXiv id.

    Используй, когда нужен текст аннотации для разбора/конспекта.
    Аргументы:
        arxiv_id: идентификатор статьи, напр. "2310.06825" или "2310.06825v1".
    Возвращает заголовок, авторов, дату, ссылку и полную аннотацию (abstract).
    """
    feed = _fetch_feed(f"id_list={quote_plus(arxiv_id)}&max_results=1")
    if getattr(feed, "status", 200) == 429:
        return (
            "arXiv временно ограничил частоту запросов (HTTP 429). "
            "Подождите минуту и повторите запрос."
        )
    if not feed.entries:
        return f"Статья с id «{arxiv_id}» не найдена."

    paper = _parse_entry(feed.entries[0])
    authors = ", ".join(paper["authors"])
    return (
        f"Заголовок: {paper['title']}\n"
        f"Авторы: {authors}\n"
        f"Дата: {paper['date']}\n"
        f"Ссылка: {paper['link']}\n\n"
        f"Аннотация:\n{paper['summary']}"
    )


# --- Безопасный калькулятор -------------------------------------------------
# Никаких eval(): разбираем выражение в AST и обходим только разрешённые узлы.
# Так модель не сможет (даже случайно) выполнить произвольный код.

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Выражение содержит недопустимые элементы.")


@tool
def calculator(expression: str) -> str:
    """Безопасно вычислить арифметическое выражение.

    Полезно для сравнения метрик из статей: разница, прирост в процентах и т.п.
    Поддерживаются + - * / ** % и скобки. Пример: "(0.97 - 0.74) / 0.74 * 100".
    Аргументы:
        expression: строка с арифметическим выражением.
    Возвращает результат вычисления.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
    except (ValueError, SyntaxError, ZeroDivisionError) as exc:
        return f"Не удалось вычислить «{expression}»: {exc}"
    return f"{expression} = {result}"
