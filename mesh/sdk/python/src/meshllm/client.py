from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

from ._binding import native
from .types import (
    Model,
    OpenAIRequestError,
    OpenAIResponse,
    OpenAIStreamChunk,
    OpenAIStreamEvent,
    OpenAIStreamStarted,
    Status,
)

_MAX_STREAM_EVENTS = 256


class _EventSink:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue: asyncio.Queue[object] = asyncio.Queue(maxsize=_MAX_STREAM_EVENTS)
        self._overflowed = False

    def on_event(self, event: object) -> None:
        self._loop.call_soon_threadsafe(self._deliver, event)

    def _deliver(self, event: object) -> None:
        if self._overflowed:
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._overflowed = True
            while not self._queue.empty():
                self._queue.get_nowait()
            self._queue.put_nowait(
                RuntimeError(
                    f"OpenAI stream consumer fell behind by {_MAX_STREAM_EVENTS} events"
                )
            )

    async def next(self) -> object:
        event = await self._queue.get()
        if isinstance(event, BaseException):
            raise event
        return event


class Inference:
    """OpenAI-compatible inference through a running embedded node."""

    def __init__(self, handle: object) -> None:
        self._handle = handle

    async def list_models(self) -> list[Model]:
        models = await asyncio.to_thread(self._handle.inference_list_models)
        return [
            Model(
                id=model.id,
                name=model.name,
                context_length=getattr(model, "context_length", None),
            )
            for model in models
        ]

    async def request(
        self,
        path: str,
        body: Mapping[str, Any],
        *,
        raise_for_status: bool = True,
    ) -> OpenAIResponse:
        """Send a lossless OpenAI-compatible request through the mesh.

        The body is serialized as-is, so agent fields such as ``tools``,
        ``tool_choice``, multimodal content blocks, ``response_format``, usage,
        and future protocol additions are not narrowed by the SDK.
        """
        response = await asyncio.to_thread(
            self._handle.openai_request,
            path,
            json.dumps(dict(body), separators=(",", ":")),
        )
        result = OpenAIResponse(
            status_code=response.status_code,
            content_type=response.content_type,
            body=response.body,
        )
        if raise_for_status and not 200 <= result.status_code < 300:
            raise OpenAIRequestError(result.status_code, result.body)
        return result

    async def chat_completions(self, body: Mapping[str, Any]) -> dict[str, Any]:
        request = dict(body)
        request["stream"] = False
        return (await self.request("/v1/chat/completions", request)).json()

    async def responses(self, body: Mapping[str, Any]) -> dict[str, Any]:
        request = dict(body)
        request["stream"] = False
        return (await self.request("/v1/responses", request)).json()

    async def stream(
        self, path: str, body: Mapping[str, Any]
    ) -> AsyncIterator[OpenAIStreamEvent]:
        """Stream complete OpenAI-compatible SSE events through the mesh.

        Each SSE payload remains unprojected. Text, reasoning, tool-call
        arguments, usage, provider extensions, and future event types are all
        available through :class:`OpenAIStreamChunk`.
        """
        request = dict(body)
        request["stream"] = True
        loop = asyncio.get_running_loop()
        sink = _EventSink(loop)
        start_task = asyncio.create_task(
            asyncio.to_thread(
                self._handle.openai_stream,
                path,
                json.dumps(request, separators=(",", ":")),
                sink,
            )
        )
        try:
            request_id = await asyncio.shield(start_task)
        except asyncio.CancelledError:
            request_id = await start_task
            await asyncio.to_thread(self._handle.cancel, request_id)
            raise
        finished = False
        try:
            while True:
                event = await sink.next()
                if event.is_started():
                    yield OpenAIStreamStarted(
                        request_id=event.request_id,
                        status_code=event.status_code,
                        content_type=event.content_type,
                    )
                elif event.is_sse():
                    yield OpenAIStreamChunk(
                        request_id=event.request_id,
                        event=event.event_type,
                        data=event.data,
                        raw=event.raw,
                    )
                elif event.is_completed():
                    finished = True
                    return
                elif event.is_failed():
                    finished = True
                    raise OpenAIRequestError(
                        event.status_code,
                        event.body,
                        message=event.error,
                    )
        finally:
            if not finished:
                await asyncio.to_thread(self._handle.cancel, request_id)

    async def stream_chat_completions(
        self, body: Mapping[str, Any]
    ) -> AsyncIterator[OpenAIStreamEvent]:
        stream = self.stream("/v1/chat/completions", body)
        try:
            async for event in stream:
                yield event
        finally:
            await stream.aclose()

    async def stream_responses(
        self, body: Mapping[str, Any]
    ) -> AsyncIterator[OpenAIStreamEvent]:
        stream = self.stream("/v1/responses", body)
        try:
            async for event in stream:
                yield event
        finally:
            await stream.aclose()


class Node:
    """An embedded mesh node in client, serve, or combined mode."""

    def __init__(self, handle: object) -> None:
        self._handle = handle
        self.inference = Inference(handle)

    @classmethod
    def create(
        cls,
        *,
        mode: str = "client",
        join_tokens: tuple[str, ...] = (),
        models: tuple[str, ...] = (),
        auto_join: bool = False,
        owner_key_path: str | None = None,
        api_port: int = 9337,
        console_port: int = 3131,
    ) -> Node:
        handle = native().create_node(
            mode,
            list(join_tokens),
            list(models),
            auto_join,
            owner_key_path,
            api_port,
            console_port,
        )
        return cls(handle)

    async def start(self) -> None:
        await asyncio.to_thread(self._handle.start)

    async def stop(self) -> None:
        await asyncio.to_thread(self._handle.stop)

    async def status(self) -> Status:
        value = await asyncio.to_thread(self._handle.status)
        return Status(
            running=value.running,
            mode=value.mode,
            api_base_url=value.api_base_url,
            console_url=value.console_url,
            payload=json.loads(value.payload_json),
        )

    async def join_token(self, token: str) -> None:
        await asyncio.to_thread(self._handle.join_token, token)

    async def __aenter__(self) -> Node:
        await self.start()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.stop()
