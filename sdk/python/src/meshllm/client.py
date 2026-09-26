from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from ._binding import native
from .types import (
    InferenceEvent,
    MeshError,
    Model,
    OpenAIRequestError,
    OpenAIResponse,
    RequestCompleted,
    Status,
    TextDelta,
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
        return [Model(id=model.id, name=model.name) for model in models]

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
        request.setdefault("stream", False)
        return (await self.request("/v1/chat/completions", request)).json()

    async def responses(self, body: Mapping[str, Any]) -> dict[str, Any]:
        request = dict(body)
        request.setdefault("stream", False)
        return (await self.request("/v1/responses", request)).json()

    async def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
    ) -> AsyncIterator[InferenceEvent]:
        """Use the typed text stream convenience API.

        Agent applications should use :meth:`chat_completions` so rich message
        and response fields remain intact.
        """
        binding = native()
        request = binding.ChatRequestNative(
            model=model,
            messages=[
                binding.ChatMessageNative(role=message["role"], content=message["content"])
                for message in messages
            ],
        )
        async for event in self._event_stream("chat", request):
            yield event

    async def text_response(self, *, model: str, input: str) -> AsyncIterator[InferenceEvent]:
        binding = native()
        request = binding.ResponsesRequestNative(model=model, input=input)
        async for event in self._event_stream("responses", request):
            yield event

    async def _event_stream(self, method: str, request: object) -> AsyncIterator[InferenceEvent]:
        loop = asyncio.get_running_loop()
        sink = _EventSink(loop)
        request_id = await asyncio.to_thread(getattr(self._handle, method), request, sink)
        finished = False
        try:
            while True:
                event = await sink.next()
                if event.is_token_delta():
                    yield TextDelta(request_id=event.request_id, text=event.delta)
                elif event.is_completed():
                    yield RequestCompleted(request_id=event.request_id)
                    finished = True
                    return
                elif event.is_failed():
                    raise MeshError(event.error)
        finally:
            if not finished:
                await asyncio.to_thread(self._handle.cancel, request_id)


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
