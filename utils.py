"""Deterministic fallbacks and length limits. No LLM calls here."""

from __future__ import annotations

import logging
import re

from schemas import (
    AnswerDraft,
    Category,
    ClassifyLabels,
    ClassifyResult,
    ExtractResult,
    SelfCheckResult,
    Sentiment,
)

logger = logging.getLogger("pipeline")

MAX_ANSWER_CHARS = 800
MAX_SUMMARY_WORDS = 40

_SENTENCE_SPLIT = re.compile(r"[.!?…\n]+")


def clip_text(text: str, limit: int) -> tuple[str, bool]:
    text = text.strip()
    if len(text) <= limit:
        return text, False
    clipped = text[: limit - 1].rstrip() + "…"
    logger.warning("clipped text %s → %s chars", len(text), len(clipped))
    return clipped, True


def fallback_extract(text: str) -> ExtractResult:
    """Heuristic extract when the model output is unusable."""
    words = text.split()
    summary = " ".join(words[:MAX_SUMMARY_WORDS]) or "Empty user message."
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    points = parts[:3]
    while len(points) < 3:
        points.append("Not specified")
    logger.warning("using fallback extract")
    return ExtractResult(summary=summary[:500], key_points=points)


def fallback_classify() -> ClassifyLabels:
    logger.warning("using fallback classify → general_question")
    return ClassifyLabels(
        category=Category.general_question,
        sentiment=Sentiment.neutral,
        intent="unclassified_fallback",
    )


def fallback_answer(text: str, packet: ClassifyResult) -> AnswerDraft:
    cyrillic = bool(re.search(r"[А-Яа-яЁё]", text))
    points = "; ".join(packet.key_points)
    if cyrillic:
        body = (
            f"Мы получили ваш запрос ({packet.intent}). "
            f"Категория: {packet.category.value}. "
            f"Кратко: {packet.summary} Ключевые детали: {points}."
        )
    else:
        body = (
            f"We received your request ({packet.intent}). "
            f"Category: {packet.category.value}. "
            f"Summary: {packet.summary} Key details: {points}."
        )
    clipped, _ = clip_text(body, MAX_ANSWER_CHARS)
    logger.warning("using fallback answer for route=%s", packet.category.value)
    return AnswerDraft(final_answer=clipped)


def skipped_self_check(reason: str) -> SelfCheckResult:
    logger.warning("self-check skipped: %s", reason)
    return SelfCheckResult(
        ok=True,
        contradicts_input=False,
        missing_details=[],
        notes=f"self-check skipped: {reason}",
    )
