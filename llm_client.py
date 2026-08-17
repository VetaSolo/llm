"""Thin wrapper around an OpenAI-compatible chat API. Retries live here, not in the pipeline."""

from __future__ import annotations

import logging
import os
import ssl
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

load_dotenv()

logger = logging.getLogger("llm")

_client: OpenAI | None = None
_script: list[Any] | None = None

MAX_RAW_CHARS = 8000


class LLMError(Exception):
    """Call failed after retries. Pipeline should fall back, not crash."""


class RetryableAPIError(Exception):
    """Transport / temporary API failure. complete() retries this."""


class EmptyReplyError(RetryableAPIError):
    """Model returned empty content."""


def _tls12_context() -> ssl.SSLContext:
    """Some Windows networks break TLS 1.3 handshakes; Groq works on TLS 1.2."""
    ctx = ssl.create_default_context()
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.maximum_version = ssl.TLSVersion.TLSv1_2
    return ctx


def get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-your-key") or api_key == "your-key-here":
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in .env."
        )

    base_url = os.getenv("OPENAI_BASE_URL") or None
    http_client = httpx.Client(
        verify=_tls12_context(),
        timeout=60.0,
        trust_env=False,
        proxy=None,
    )
    _client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        timeout=60.0,
    )
    return _client


def _max_retries() -> int:
    return max(1, int(os.getenv("LLM_MAX_RETRIES", "3")))


def _retry_base() -> float:
    return float(os.getenv("LLM_RETRY_BASE", "0.5"))


@contextmanager
def scripted(events: Sequence[Any]) -> Iterator[None]:
    """Replay canned replies/errors instead of calling the API. Used by --demo-failures."""
    global _script
    _script = list(events)
    try:
        yield
    finally:
        _script = None


def _once(
    prompt: str,
    *,
    system: str | None,
    temperature: float,
    json_mode: bool,
) -> str:
    if _script is not None:
        if not _script:
            raise LLMError("LLM script exhausted")
        event = _script.pop(0)
        if isinstance(event, BaseException):
            raise event
        return str(event)

    client = get_client()
    model = os.getenv("OPENAI_MODEL", "llama-3.3-70b-versatile")
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    return (content or "").strip()


def complete(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.2,
    json_mode: bool = False,
) -> str:
    """Send chat messages. Retries temporary API errors and empty replies."""
    retries = _max_retries()
    last: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            content = _once(
                prompt,
                system=system,
                temperature=temperature,
                json_mode=json_mode,
            )
            if not content.strip():
                raise EmptyReplyError("empty model response")
            if len(content) > MAX_RAW_CHARS:
                logger.warning("raw model output truncated %s → %s chars", len(content), MAX_RAW_CHARS)
                content = content[:MAX_RAW_CHARS]
            if attempt > 1:
                logger.info("LLM succeeded on attempt %s/%s", attempt, retries)
            return content
        except LLMError:
            raise
        except (RetryableAPIError, APITimeoutError, APIConnectionError, RateLimitError) as exc:
            last = exc
            logger.warning("LLM attempt %s/%s failed (retryable): %s", attempt, retries, exc)
        except APIStatusError as exc:
            last = exc
            if exc.status_code and exc.status_code >= 500:
                logger.warning("LLM attempt %s/%s failed (HTTP %s)", attempt, retries, exc.status_code)
            else:
                logger.error("LLM non-retryable HTTP error: %s", exc)
                raise LLMError(f"LLM HTTP error: {exc}") from exc
        except Exception as exc:
            last = exc
            logger.exception("LLM unexpected error")
            raise LLMError(f"LLM unexpected error: {exc}") from exc

        if attempt < retries:
            delay = _retry_base() * (2 ** (attempt - 1))
            logger.info("retrying in %.2fs", delay)
            time.sleep(delay)

    raise LLMError(f"LLM call failed after {retries} attempts: {last}") from last
