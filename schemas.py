"""Pydantic contracts for each pipeline step."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE)
T = TypeVar("T", bound=BaseModel)


class Category(str, Enum):
    support = "support"
    feedback = "feedback"
    complaint = "complaint"
    sales = "sales"
    general_question = "general_question"


class Sentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


def _strip(value: object) -> object:
    return value.strip() if isinstance(value, str) else value


def _three_points(value: list[str]) -> list[str]:
    cleaned = [item.strip() for item in value if str(item).strip()]
    if len(cleaned) != 3:
        raise ValueError("key_points must contain exactly 3 non-empty strings")
    return cleaned


class ExtractResult(BaseModel):
    """Step 1: meaning only. No routing labels yet."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(min_length=1, max_length=500)
    key_points: list[str] = Field(min_length=3, max_length=3)

    @field_validator("summary", mode="before")
    @classmethod
    def strip_summary(cls, value: object) -> object:
        return _strip(value)

    @field_validator("key_points")
    @classmethod
    def clean_points(cls, value: list[str]) -> list[str]:
        return _three_points(value)


class ClassifyLabels(BaseModel):
    """Step 2: routing keys. Uses extract output as context."""

    model_config = ConfigDict(extra="ignore")

    category: Category
    sentiment: Sentiment
    intent: str = Field(min_length=1, max_length=80)

    @field_validator("category", "sentiment", mode="before")
    @classmethod
    def normalize_enum(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("intent", mode="before")
    @classmethod
    def strip_intent(cls, value: object) -> object:
        return _strip(value)

    @field_validator("intent")
    @classmethod
    def normalize_intent(cls, value: str) -> str:
        return value.lower().replace(" ", "_")


class ClassifyResult(BaseModel):
    """Step 3: structured packet assembled in code from steps 1 and 2."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(min_length=1, max_length=500)
    category: Category
    sentiment: Sentiment
    intent: str = Field(min_length=1, max_length=80)
    key_points: list[str] = Field(min_length=3, max_length=3)

    @field_validator("summary", "intent", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        return _strip(value)

    @field_validator("category", "sentiment", mode="before")
    @classmethod
    def normalize_enum(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("intent")
    @classmethod
    def normalize_intent(cls, value: str) -> str:
        return value.lower().replace(" ", "_")

    @field_validator("key_points")
    @classmethod
    def clean_points(cls, value: list[str]) -> list[str]:
        return _three_points(value)


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    final_answer: str = Field(min_length=1, max_length=1500)

    @field_validator("final_answer", mode="before")
    @classmethod
    def strip_answer(cls, value: object) -> object:
        return _strip(value)


class SelfCheckResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ok: bool
    contradicts_input: bool
    missing_details: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("notes", mode="before")
    @classmethod
    def strip_notes(cls, value: object) -> object:
        return _strip(value) if value is not None else ""

    @field_validator("missing_details")
    @classmethod
    def clean_missing(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if str(item).strip()]

    @model_validator(mode="after")
    def fold_ok(self) -> SelfCheckResult:
        if self.contradicts_input or self.missing_details:
            return self.model_copy(update={"ok": False})
        return self


class StepEvent(BaseModel):
    name: str
    uses: str
    status: str
    detail: str


class PipelineResult(ClassifyResult):
    final_answer: str = Field(min_length=1, max_length=1500)
    route: str = Field(min_length=1)
    revised: bool = False
    self_check: SelfCheckResult
    steps: list[StepEvent] = Field(default_factory=list)
    degraded: bool = False
    fallback_used: list[str] = Field(default_factory=list)
    truncated: bool = False


class StructuredParseError(Exception):
    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


def extract_json_object(raw: str) -> dict:
    text = FENCE_RE.sub("", raw.strip()).strip()
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(data, dict):
            return data
        last_error = TypeError(f"JSON root must be an object, got {type(data).__name__}")

    preview = raw.strip().replace("\n", " ")[:180]
    reason = str(last_error) if last_error else "no JSON object found"
    raise StructuredParseError(
        f"Model did not return valid JSON ({reason}). Preview: {preview}",
        raw=raw,
    )


def parse_model(raw: str, model_type: type[T]) -> T:
    data = extract_json_object(raw)
    try:
        return model_type.model_validate(data)
    except ValidationError as exc:
        raise StructuredParseError(_format_validation_error(exc), raw=raw) from exc


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        location = ".".join(str(item) for item in err.get("loc", ())) or "root"
        parts.append(f"{location}: {err.get('msg')}")
    return "JSON parsed, but schema validation failed: " + "; ".join(parts)
