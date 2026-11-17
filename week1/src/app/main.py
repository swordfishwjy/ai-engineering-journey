import logging
from collections.abc import AsyncIterable
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated, Literal

import anthropic
from fastapi import FastAPI, Header, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field

from .config import get_settings
from .llm import RETRYABLE, Done, Fault, TextDelta, make_client, stream_chat

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.client = make_client(settings)
    yield
    await app.state.client.close()


app = FastAPI(lifespan=lifespan)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    system: str | None = None
    max_tokens: int = Field(default=4096, ge=1, le=64000)


def error_event(err: anthropic.APIError) -> ServerSentEvent:
    retryable = isinstance(err, RETRYABLE)
    return ServerSentEvent(
        event="error",
        data={
            "type": type(err).__name__,
            "status": getattr(err, "status_code", None),
            "retryable": retryable,
            "message": "Upstream busy, retry later" if retryable else "Upstream request failed",
        },
    )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/stream", response_class=EventSourceResponse)
async def chat_stream(
    body: ChatRequest,
    request: Request,
    x_simulate_fault: Annotated[Fault | None, Header()] = None,
    x_simulate_fault_count: Annotated[int, Header(ge=0)] = 1,
) -> AsyncIterable[ServerSentEvent]:
    settings = request.app.state.settings
    fault = x_simulate_fault if settings.enable_fault_injection else None
    try:
        async for item in stream_chat(
            request.app.state.client,
            settings,
            messages=[m.model_dump() for m in body.messages],
            system=body.system,
            max_tokens=body.max_tokens,
            fault=fault,
            fault_count=x_simulate_fault_count,
        ):
            match item:
                case TextDelta(text=text):
                    yield ServerSentEvent(event="delta", data={"text": text})
                case Done(stop_reason="refusal"):
                    yield ServerSentEvent(
                        event="error",
                        data={"type": "refusal", "retryable": False, "message": "Request declined"},
                    )
                case Done():
                    yield ServerSentEvent(event="done", data=asdict(item))
    except anthropic.APIError as err:
        log.exception("chat stream failed")
        yield error_event(err)
