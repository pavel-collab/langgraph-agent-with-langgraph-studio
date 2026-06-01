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
"""

from __future__ import annotations

import ast
import operator

import arxiv
from langchain_core.tools import tool

# Один общий клиент arxiv на модуль — он сам управляет паузами между запросами.
# page_size=5: не тянем по 100 записей на страницу (так запрос «легче» для
#   рейт-лимитера arXiv и в URL уходит max_results=5, а не 100).
# delay_seconds=3.0: arXiv просит не чаще одного запроса в 3 с.
# num_retries=5: при временном 429/5xx клиент сам повторит с паузой.
_arxiv_client = arxiv.Client(page_size=5, delay_seconds=3.0, num_retries=5)


@tool
def search_arxiv(query: str, max_results: int = 5) -> str:
    """Найти научные статьи на arXiv по ключевым словам.

    Используй для запросов вида «найди статьи про…», «что есть по теме…».
    Аргументы:
        query: поисковый запрос на английском (напр. "graph retrieval augmented generation").
        max_results: сколько статей вернуть (по умолчанию 5).
    Возвращает список статей: arXiv id, заголовок, авторы, дата и ссылка.
    """
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    try:
        results = list(_arxiv_client.results(search))
    except arxiv.HTTPError as exc:
        # 429 = arXiv троттлит наш IP (слишком частые запросы). Не роняем граф —
        # возвращаем агенту понятный текст, чтобы он мог ответить пользователю.
        if exc.status == 429:
            return (
                "arXiv временно ограничил частоту запросов (HTTP 429). "
                "Подождите минуту и повторите запрос."
            )
        return f"Ошибка обращения к arXiv (HTTP {exc.status}). Повторите позже."
    if not results:
        return f"По запросу «{query}» ничего не найдено."

    lines = []
    for paper in results:
        arxiv_id = paper.get_short_id()
        authors = ", ".join(a.name for a in paper.authors[:3])
        if len(paper.authors) > 3:
            authors += " и др."
        date = paper.published.strftime("%Y-%m-%d")
        article_summary = paper.summary.replace("\n", " ").strip()
        lines.append(
            f"- [{arxiv_id}] {paper.title}\n"
            f"  Авторы: {authors}\n"
            f"  Дата: {date}\n"
            f"  Кратко: {article_summary[:200]}...\n"
            f"  Ссылка: {paper.entry_id}"
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
    search = arxiv.Search(id_list=[arxiv_id])
    try:
        paper = next(_arxiv_client.results(search))
    except StopIteration:
        return f"Статья с id «{arxiv_id}» не найдена."
    except arxiv.HTTPError as exc:
        if exc.status == 429:
            return (
                "arXiv временно ограничил частоту запросов (HTTP 429). "
                "Подождите минуту и повторите запрос."
            )
        return f"Ошибка обращения к arXiv (HTTP {exc.status}). Повторите позже."

    authors = ", ".join(a.name for a in paper.authors)
    date = paper.published.strftime("%Y-%m-%d")
    return (
        f"Заголовок: {paper.title}\n"
        f"Авторы: {authors}\n"
        f"Дата: {date}\n"
        f"Ссылка: {paper.entry_id}\n\n"
        f"Аннотация:\n{paper.summary}"
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
