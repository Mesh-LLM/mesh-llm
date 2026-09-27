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


class _EventSink:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue: asyncio.Queue[object] = asyncio.Queue()

    def on_event(self, event: object) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    async def next(self) -> object:
        return await self._queue.get()


class Inference:
    """Inference APIs shared by :class:`Client` and :class:`Node`."""

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
        request_id = await asyncio.to_thread(
            self._handle.openai_stream,
            path,
            json.dumps(request, separators=(",", ":")),
            sink,
        )
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


class _Lifecycle:
    def __init__(self, handle: object) -> None:
        self._handle = handle
        self.inference = Inference(handle)

    async def start(self) -> None:
        await asyncio.to_thread(self._handle.start)

    async def stop(self) -> None:
        await asyncio.to_thread(self._handle.stop)

    async def reconnect(self) -> None:
        await asyncio.to_thread(self._handle.reconnect)

    async def status(self) -> Status:
        status = await asyncio.to_thread(self._handle.status)
        return Status(connected=status.connected, peer_count=status.peer_count)

    async def __aenter__(self) -> _Lifecycle:
        await self.start()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.stop()


class Client(_Lifecycle):
    """Client-only connection to an existing public or private mesh."""

    @classmethod
    async def connect_public(
        cls,
        *,
        owner_keypair_hex: str,
        model: str | None = None,
        min_vram_gb: float | None = None,
        region: str | None = None,
        target_name: str | None = None,
        relays: tuple[str, ...] = (),
    ) -> Client:
        """Discover and connect to the best matching published mesh."""
        binding = native()
        query = binding.PublicMeshQuery(
            model=model,
            min_vram_gb=min_vram_gb,
            region=region,
            target_name=target_name,
            relays=list(relays),
        )
        handle = await asyncio.to_thread(
            binding.create_auto_client,
            owner_keypair_hex,
            query,
        )
        return cls(handle)

    @classmethod
    def create(cls, *, owner_keypair_hex: str, invite_token: str) -> Client:
        handle = native().create_client(owner_keypair_hex, invite_token)
        return cls(handle)


class Node(_Lifecycle):
    """A mesh client that can also manage and serve local models."""

    @classmethod
    def create(
        cls,
        *,
        owner_keypair_hex: str,
        invite_token: str,
        cache_dir: str | None = None,
        runtime_dir: str | None = None,
        serving_enabled: bool = False,
    ) -> Node:
        handle = native().create_node(
            owner_keypair_hex,
            invite_token,
            cache_dir,
            runtime_dir,
            serving_enabled,
        )
        return cls(handle)
