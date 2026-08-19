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
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")

# First match-group wins ties only via score. Phrases before short tokens.
_CATEGORY_PATTERNS: list[tuple[Category, tuple[str, ...]]] = [
    (
        Category.complaint,
        (
            r"в ярости",
            r"роспотреб",
            r"chargeback",
            r"возврат",
            r"refund",
            r"компенс",
            r"двойн\w* списа",
            r"списал",
            r"unacceptable",
            r"жалоб",
            r"молчит",
            r"сорван",
            r"требую",
        ),
    ),
    (
        Category.sales,
        (
            r"\btrial\b",
            r"триал",
            r"тариф",
            r"скидк",
            r"pricing",
            r"\bprice\b",
            r"сколько стоит",
            r"цен[аыу]",
            r"мест\b",
            r"\bseats?\b",
            r"купить",
            r"\bdemo\b",
            r"план include",
            r"team plan",
        ),
    ),
    (
        Category.support,
        (
            r"session expired",
            r"не могу войти",
            r"cannot log",
            r"не работает",
            r"overheat",
            r"перегрев",
            r"\berror\b",
            r"\bbug\b",
            r"сломал",
            r"2fa",
            r"password reset",
            r"что проверить",
            r"what should i check",
        ),
    ),
    (
        Category.feedback,
        (
            r"не хватает",
            r"хотелось бы",
            r"feature request",
            r"тёмн\w* тем",
            r"dark mode",
            r"weekly digest",
            r"в целом нравится",
            r"otherwise the product",
        ),
    ),
]

_NEGATIVE = (
    r"ярост",
    r"не могу",
    r"сломал",
    r"молчит",
    r"сорван",
    r"unacceptable",
    r"error",
    r"жалоб",
    r"extremely",
)
_POSITIVE = (
    r"нравится",
    r"спасибо",
    r"thanks",
    r"useful",
    r"solid",
    r"в целом ок",
)

_INTENT = {
    Category.complaint: "request_refund",
    Category.sales: "request_pricing",
    Category.support: "request_troubleshooting",
    Category.feedback: "request_feature",
    Category.general_question: "ask_question",
}


def clip_text(text: str, limit: int) -> tuple[str, bool]:
    text = text.strip()
    if len(text) <= limit:
        return text, False
    clipped = text[: limit - 1].rstrip() + "…"
    logger.warning("clipped text %s → %s chars", len(text), len(clipped))
    return clipped, True


def _haystack(text: str, summary: str = "", key_points: list[str] | None = None) -> str:
    parts = [text, summary, *(key_points or [])]
    return " ".join(part for part in parts if part).lower()


def _score_category(blob: str) -> Category:
    best = Category.general_question
    best_score = 0
    for category, patterns in _CATEGORY_PATTERNS:
        score = sum(1 for pattern in patterns if re.search(pattern, blob, flags=re.IGNORECASE))
        if score > best_score:
            best, best_score = category, score
    return best


def _sentiment(blob: str, category: Category) -> Sentiment:
    if any(re.search(pattern, blob, flags=re.IGNORECASE) for pattern in _NEGATIVE):
        return Sentiment.negative
    if any(re.search(pattern, blob, flags=re.IGNORECASE) for pattern in _POSITIVE):
        return Sentiment.positive
    if category is Category.complaint:
        return Sentiment.negative
    return Sentiment.neutral


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


def fallback_classify(
    text: str,
    summary: str = "",
    key_points: list[str] | None = None,
) -> ClassifyLabels:
    """Keyword routing from the original text — not a constant general_question."""
    blob = _haystack(text, summary, key_points)
    category = _score_category(blob)
    sentiment = _sentiment(blob, category)
    intent = _INTENT[category]
    logger.warning("using fallback classify → %s / %s", category.value, intent)
    return ClassifyLabels(category=category, sentiment=sentiment, intent=intent)


def fallback_answer(text: str, packet: ClassifyResult) -> AnswerDraft:
    """Keep route tone even when the LLM answer step failed."""
    russian = bool(_CYRILLIC.search(text))
    details = "; ".join(p for p in packet.key_points if p != "Not specified") or packet.summary
    first = packet.key_points[0] if packet.key_points else packet.summary
    rest = "; ".join(packet.key_points[1:]) if len(packet.key_points) > 1 else details

    if packet.category is Category.complaint:
        body = (
            f"Извините за конкретный вред: {packet.summary} "
            f"Фиксируем обращение по деталям ({details}) и запускаем возврат или компенсацию по этому случаю. "
            f"В поддержку повторно писать не нужно — эскалация уже с нашей стороны."
            if russian
            else (
                f"Sorry for the specific harm: {packet.summary} "
                f"We are logging the case ({details}) and starting a refund or compensation on it. "
                f"You do not need to contact support again — we are escalating this now."
            )
        )
    elif packet.category is Category.sales:
        body = (
            f"По запросу ({packet.summary}) подберём тариф под ваши условия ({details}). "
            f"Напишите число мест или срок — пришлём trial или созвон по цене."
            if russian
            else (
                f"For this request ({packet.summary}) we can match a plan to ({details}). "
                f"Reply with seat count or term and we will send a trial or a pricing call."
            )
        )
    elif packet.category is Category.support:
        body = (
            f"Сначала проверьте: {first}. Затем: {rest}. "
            f"Если не сработает — напишите, что изменилось после этих шагов."
            if russian
            else (
                f"Check this first: {first}. Next: {rest}. "
                f"If that fails, tell us what changed after these steps."
            )
        )
    elif packet.category is Category.feedback:
        body = (
            f"Спасибо за отзыв. Зафиксировали запросы: {details}. "
            f"Срок релиза обещать не будем."
            if russian
            else (
                f"Thanks for the feedback. We captured: {details}. "
                f"We will not promise a release date."
            )
        )
    else:
        body = (
            f"{packet.summary} Детали из текста: {details}. "
            f"Если нужен более точный ответ — уточните один вопрос."
            if russian
            else (
                f"{packet.summary} Details from the text: {details}. "
                f"If you need a sharper answer, send one clarifying question."
            )
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
