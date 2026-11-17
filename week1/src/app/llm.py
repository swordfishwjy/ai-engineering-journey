import asyncio
import logging
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

import anthropic
import httpx2

from .config import Settings

log = logging.getLogger(__name__)

# Same set the SDK retries internally: connection/timeout, 408, 409, 429, 5xx (incl. 529).
RETRYABLE = (
    anthropic.APIConnectionError,
    anthropic.RequestTimeoutError,
    anthropic.ConflictError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)

Fault = Literal["429", "500", "timeout"]


@dataclass
class TextDelta:
    text: str


@dataclass
class Done:
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    attempts: int


def make_client(settings: Settings) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key.get_secret_value(),
        timeout=settings.request_timeout,
        max_retries=0,  # we own retries; SDK default (2) x ours would multiply
    )


def backoff_delay(attempt: int, err: Exception, settings: Settings) -> float:
    if isinstance(err, anthropic.APIStatusError):
        retry_after = err.response.headers.get("retry-after")
        try:
            return min(float(retry_after), settings.retry_max_delay)
        except (TypeError, ValueError):
            pass
    # "full jitter": uniform(0, min(cap, base * 2^n))
    return random.uniform(0, min(settings.retry_max_delay, settings.retry_base_delay * 2**attempt))


_FAKE_REQ = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def simulated_error(fault: Fault) -> anthropic.APIError:
    match fault:
        case "429":
            resp = httpx2.Response(429, request=_FAKE_REQ, headers={"retry-after": "1"})
            return anthropic.RateLimitError("simulated 429", response=resp, body=None)
        case "500":
            resp = httpx2.Response(500, request=_FAKE_REQ)
            return anthropic.InternalServerError("simulated 500", response=resp, body=None)
        case "timeout":
            return anthropic.APITimeoutError(request=_FAKE_REQ)


async def stream_chat(
    client: anthropic.AsyncAnthropic,
    settings: Settings,
    *,
    messages: list[dict[str, Any]],
    system: str | None,
    max_tokens: int,
    fault: Fault | None = None,
    fault_count: int = 0,
) -> AsyncIterator[TextDelta | Done]:
    extra = {"system": system} if system else {}
    for attempt in range(settings.max_retries + 1):
        emitted = False
        try:
            if fault and attempt < fault_count:
                raise simulated_error(fault)
            async with client.beta.messages.stream(
                model=settings.default_model,
                max_tokens=max_tokens,
                messages=messages,
                output_config={"effort": settings.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",  # re-run on a fallback model if a safety classifier declines
                **extra,
            ) as stream:
                async for text in stream.text_stream:
                    emitted = True
                    yield TextDelta(text)
                final = await stream.get_final_message()
            yield Done(
                stop_reason=final.stop_reason,
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
                attempts=attempt + 1,
            )
            return
        except RETRYABLE as err:
            # After the first token the client already has partial text: don't silently restart.
            if emitted or attempt == settings.max_retries:
                raise
            delay = backoff_delay(attempt, err, settings)
            log.warning("attempt %d failed (%s); retrying in %.2fs", attempt + 1, type(err).__name__, delay)
            await asyncio.sleep(delay)
