"""Multi-step chain with retries (in the client) and per-step fallbacks."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from llm_client import LLMError, complete
from prompts import CLASSIFY, EXTRACT, REPAIR, REVISE, SELF_CHECK, PromptSpec, route_for
from schemas import (
    AnswerDraft,
    Category,
    ClassifyLabels,
    ClassifyResult,
    ExtractResult,
    PipelineResult,
    SelfCheckResult,
    StepEvent,
    StructuredParseError,
    parse_model,
)
from utils import (
    MAX_ANSWER_CHARS,
    clip_text,
    fallback_answer,
    fallback_classify,
    fallback_extract,
    skipped_self_check,
)

logger = logging.getLogger("pipeline")


@dataclass
class Outcome:
    source: str
    input_text: str
    ok: bool
    result: PipelineResult | None
    error: str | None
    forced_category: Category | None = None
    steps: list[StepEvent] = field(default_factory=list)
    degraded: bool = False
    fallback_used: list[str] = field(default_factory=list)


def _points(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _step(steps: list[StepEvent], name: str, uses: str, status: str, detail: str) -> None:
    event = StepEvent(name=name, uses=uses, status=status, detail=detail)
    steps.append(event)
    log = logger.warning if status in {"fail", "fallback"} else logger.info
    log("%s [%s] uses=%s | %s", name, status, uses, detail)


def _llm(spec: PromptSpec, model_type, **fields: str):
    raw = complete(
        spec.build_user(**fields),
        system=spec.system,
        json_mode=spec.json_mode,
    )
    return parse_model(raw, model_type), raw


def _repair(model_type, raw: str):
    schema = json.dumps(model_type.model_json_schema(), ensure_ascii=False)
    repaired = complete(
        REPAIR.build_user(schema=schema, raw=raw or ""),
        system=REPAIR.system,
        json_mode=True,
    )
    return parse_model(repaired, model_type), repaired


def _call_step(spec: PromptSpec, model_type, fallback, **fields: str):
    """Try the prompt, then JSON repair, then a deterministic fallback."""
    try:
        parsed, raw = _llm(spec, model_type, **fields)
        return parsed, raw, False
    except StructuredParseError as exc:
        logger.warning("invalid model output, trying JSON repair: %s", exc)
        try:
            parsed, raw = _repair(model_type, exc.raw)
            logger.info("JSON repair succeeded for %s", spec.key)
            return parsed, raw, False
        except (StructuredParseError, LLMError) as repair_exc:
            logger.warning("JSON repair failed (%s), using fallback for %s", repair_exc, spec.key)
            return fallback(), exc.raw, True
    except LLMError as exc:
        logger.warning("API failed on %s (%s), using fallback", spec.key, exc)
        return fallback(), "", True


def run_pipeline(
    text: str,
    *,
    source: str = "cli",
    force_category: Category | None = None,
) -> Outcome:
    steps: list[StepEvent] = []
    fallbacks: list[str] = []
    text = text.strip()

    try:
        return _run(text, source, force_category, steps, fallbacks)
    except Exception as exc:
        logger.exception("pipeline crashed unexpectedly")
        _step(steps, "pipeline", "internal", "fail", str(exc))
        return Outcome(source, text, False, None, str(exc), force_category, steps, True, fallbacks)


def _run(
    text: str,
    source: str,
    force_category: Category | None,
    steps: list[StepEvent],
    fallbacks: list[str],
) -> Outcome:
    extracted, _, fb = _call_step(
        EXTRACT,
        ExtractResult,
        lambda: fallback_extract(text),
        user_text=text,
    )
    if fb:
        fallbacks.append("extract")
        _step(steps, "extract", "raw text", "fallback", f"heuristic extract: {extracted.summary!r}")
    else:
        _step(
            steps,
            "extract",
            "raw text",
            "ok",
            f"summary={extracted.summary!r}; points={extracted.key_points}",
        )

    labels, _, fb = _call_step(
        CLASSIFY,
        ClassifyLabels,
        fallback_classify,
        user_text=text,
        summary=extracted.summary,
        key_points=_points(extracted.key_points),
    )
    if fb:
        fallbacks.append("classify")
        _step(steps, "classify", "extract + raw text", "fallback", f"{labels.category.value} / {labels.intent}")
    else:
        _step(
            steps,
            "classify",
            "extract + raw text",
            "ok",
            f"{labels.category.value} / {labels.intent} / {labels.sentiment.value}",
        )

    packet = ClassifyResult(
        summary=extracted.summary,
        key_points=extracted.key_points,
        category=labels.category,
        sentiment=labels.sentiment,
        intent=labels.intent,
    )
    _step(steps, "structure", "extract + classify", "ok", "assembled ClassifyResult in code")

    routed_category = force_category or packet.category
    route = route_for(routed_category)
    packet_for_reply = packet.model_copy(update={"category": routed_category})
    _step(
        steps,
        "route",
        "classify.category",
        "ok",
        f"{route.key}" + (" (forced)" if force_category else ""),
    )

    draft, _, fb = _call_step(
        route,
        AnswerDraft,
        lambda: fallback_answer(text, packet_for_reply),
        user_text=text,
        category=packet_for_reply.category.value,
        intent=packet.intent,
        sentiment=packet.sentiment.value,
        summary=packet.summary,
        key_points=_points(packet.key_points),
    )
    if fb:
        fallbacks.append("answer")
        _step(steps, "answer", f"structure + route:{route.key}", "fallback", draft.final_answer[:180])
    else:
        _step(steps, "answer", f"structure + route:{route.key}", "ok", draft.final_answer[:180])

    check, _, fb = _call_step(
        SELF_CHECK,
        SelfCheckResult,
        lambda: skipped_self_check("api or invalid JSON"),
        user_text=text,
        category=packet_for_reply.category.value,
        intent=packet.intent,
        summary=packet.summary,
        key_points=_points(packet.key_points),
        final_answer=draft.final_answer,
    )
    if fb:
        fallbacks.append("self_check")
        _step(steps, "self_check", "answer + raw text", "fallback", check.notes)
    else:
        missing = ", ".join(check.missing_details) or "—"
        _step(
            steps,
            "self_check",
            "answer + raw text",
            "ok" if check.ok else "fail",
            f"ok={check.ok} contradicts={check.contradicts_input} missing=[{missing}] notes={check.notes!r}",
        )

    revised = False
    final_answer = draft.final_answer
    if not check.ok and "self_check" not in fallbacks:
        try:
            fixed, _, fb = _call_step(
                REVISE,
                AnswerDraft,
                lambda: draft,
                user_text=text,
                route=route.key,
                final_answer=draft.final_answer,
                notes=check.notes or "—",
                contradicts_input=str(check.contradicts_input).lower(),
                missing_details=_points(check.missing_details) if check.missing_details else "- (none listed)",
            )
            if fb:
                _step(steps, "revise", "self_check", "fallback", "kept original draft")
            else:
                final_answer = fixed.final_answer
                revised = True
                _step(steps, "revise", "self_check", "ok", final_answer[:180])
        except Exception as exc:
            logger.warning("revise failed, keeping draft: %s", exc)
            _step(steps, "revise", "self_check", "fallback", str(exc))

    final_answer, truncated = clip_text(final_answer, MAX_ANSWER_CHARS)
    if truncated:
        fallbacks.append("truncate")
        _step(steps, "guardrail", "final_answer", "ok", f"clipped to {MAX_ANSWER_CHARS} chars")

    result = PipelineResult(
        **packet_for_reply.model_dump(),
        final_answer=final_answer,
        route=route.key,
        revised=revised,
        self_check=check,
        steps=steps,
        degraded=bool(fallbacks),
        fallback_used=fallbacks,
        truncated=truncated,
    )
    return Outcome(
        source,
        text,
        True,
        result,
        None,
        force_category,
        steps,
        degraded=bool(fallbacks),
        fallback_used=fallbacks,
    )
