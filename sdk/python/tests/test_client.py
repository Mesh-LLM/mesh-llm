from __future__ import annotations

import json
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))

from meshllm import (
    Client,
    OpenAIRequestError,
    OpenAIStreamChunk,
    OpenAIStreamStarted,
)


class FakeOpenAIStreamEvent:
    def __init__(self, kind: str, **values: object) -> None:
        self.kind = kind
        self.request_id = "stream-1"
        self.status_code = values.get("status_code")
        self.content_type = values.get("content_type")
        self.event_type = values.get("event_type")
        self.data = values.get("data")
        self.raw = values.get("raw")
        self.error = values.get("error")
        self.body = values.get("body")

    def is_started(self) -> bool:
        return self.kind == "started"

    def is_sse(self) -> bool:
        return self.kind == "sse"

    def is_completed(self) -> bool:
        return self.kind == "completed"

    def is_failed(self) -> bool:
        return self.kind == "failed"


class FakeHandle:
    def __init__(self) -> None:
        self.started = False
        self.cancelled: list[str] = []
        self.last_openai: tuple[str, dict[str, object]] | None = None

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def reconnect(self) -> None:
        self.started = True

    def status(self) -> object:
        return SimpleNamespace(connected=self.started, peer_count=2)

    def inference_list_models(self) -> list[object]:
        return [SimpleNamespace(id="model-a", name="Model A", context_length=131_072)]

    def openai_request(self, path: str, body_json: str) -> object:
        body = json.loads(body_json)
        self.last_openai = (path, body)
        if body.get("model") == "missing":
            return SimpleNamespace(status_code=404, content_type="application/json", body='{"error":"missing"}')
        response = {
            "choices": [{
                "message": {"tool_calls": body.get("tools", [])},
                "finish_reason": "tool_calls",
            }],
            "usage": {"total_tokens": 12},
        }
        return SimpleNamespace(status_code=200, content_type="application/json", body=json.dumps(response))

    def openai_stream(self, path: str, body_json: str, listener: object) -> str:
        body = json.loads(body_json)
        self.last_openai = (path, body)
        if body.get("model") == "missing":
            listener.on_event(
                FakeOpenAIStreamEvent(
                    "failed",
                    status_code=404,
                    error="model not found",
                    body='{"error":"missing"}',
                )
            )
            return "stream-1"
        listener.on_event(FakeOpenAIStreamEvent(
            "started", status_code=200, content_type="text/event-stream"
        ))
        tool_delta = {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": '{"city":"Syd'},
                    }],
                },
            }],
        }
        data = json.dumps(tool_delta)
        listener.on_event(FakeOpenAIStreamEvent(
            "sse",
            event_type=(
                "response.function_call_arguments.delta"
                if path == "/v1/responses"
                else None
            ),
            data=data,
            raw=f"data: {data}\n\n",
        ))
        listener.on_event(FakeOpenAIStreamEvent("sse", data="[DONE]", raw="data: [DONE]\n\n"))
        listener.on_event(FakeOpenAIStreamEvent("completed"))
        return "stream-1"

    def cancel(self, request_id: str) -> None:
        self.cancelled.append(request_id)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_connect_public_discovers_with_query(self) -> None:
        handle = FakeHandle()
        binding = SimpleNamespace(
            PublicMeshQuery=lambda **values: SimpleNamespace(**values),
            create_auto_client=lambda owner, query: (
                self.assertEqual(owner, "ab" * 32),
                self.assertEqual(query.target_name, "community"),
                self.assertEqual(query.relays, ["wss://relay.example"]),
                handle,
            )[-1],
        )

        with patch("meshllm.client.native", return_value=binding):
            client = await Client.connect_public(
                owner_keypair_hex="ab" * 32,
                target_name="community",
                relays=("wss://relay.example",),
            )

        self.assertIs(client._handle, handle)

    async def test_lifecycle_and_models(self) -> None:
        handle = FakeHandle()
        client = Client(handle)

        async with client:
            self.assertTrue((await client.status()).connected)
            model = (await client.inference.list_models())[0]
            self.assertEqual(model.id, "model-a")
            self.assertEqual(model.context_length, 131_072)

        self.assertFalse(handle.started)

    async def test_agent_request_preserves_tools_and_full_response(self) -> None:
        handle = FakeHandle()
        client = Client(handle)
        tool = {"type": "function", "function": {"name": "search"}}

        result = await client.inference.chat_completions({
            "model": "model-a",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "find it"}]}],
            "tools": [tool],
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer"}},
            "stream": True,
        })

        self.assertEqual(handle.last_openai[0], "/v1/chat/completions")
        self.assertEqual(handle.last_openai[1]["tools"], [tool])
        self.assertFalse(handle.last_openai[1]["stream"])
        self.assertEqual(result["choices"][0]["message"]["tool_calls"], [tool])
        self.assertEqual(result["usage"]["total_tokens"], 12)

    async def test_non_success_response_raises_typed_error(self) -> None:
        client = Client(FakeHandle())

        with self.assertRaises(OpenAIRequestError) as raised:
            await client.inference.chat_completions({"model": "missing", "messages": []})

        self.assertEqual(raised.exception.status_code, 404)

    async def test_agent_stream_preserves_tool_call_deltas_and_raw_sse(self) -> None:
        handle = FakeHandle()
        events = [
            event
            async for event in Client(handle).inference.stream_chat_completions({
                "model": "model-a",
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [{"type": "function", "function": {"name": "weather"}}],
            })
        ]

        self.assertIsInstance(events[0], OpenAIStreamStarted)
        self.assertIsInstance(events[1], OpenAIStreamChunk)
        self.assertEqual(
            events[1].json()["choices"][0]["delta"]["tool_calls"][0]["index"],
            0,
        )
        self.assertIn("data:", events[1].raw)
        self.assertTrue(events[2].done)
        self.assertTrue(handle.last_openai[1]["stream"])

    async def test_responses_stream_preserves_named_events(self) -> None:
        handle = FakeHandle()
        events = [
            event
            async for event in Client(handle).inference.stream_responses({
                "model": "model-a",
                "input": "weather?",
                "tools": [{"type": "function", "name": "weather"}],
            })
        ]

        self.assertEqual(handle.last_openai[0], "/v1/responses")
        self.assertEqual(events[1].event, "response.function_call_arguments.delta")

    async def test_stream_failure_raises_typed_error_with_http_context(self) -> None:
        stream = Client(FakeHandle()).inference.stream_chat_completions({
            "model": "missing",
            "messages": [],
        })

        with self.assertRaises(OpenAIRequestError) as raised:
            await anext(stream)

        self.assertEqual(raised.exception.status_code, 404)
        self.assertEqual(raised.exception.body, '{"error":"missing"}')

    async def test_closing_agent_stream_cancels_native_request(self) -> None:
        handle = FakeHandle()
        stream = Client(handle).inference.stream_chat_completions({
            "model": "model-a",
            "messages": [{"role": "user", "content": "weather?"}],
        })

        first = await anext(stream)
        self.assertIsInstance(first, OpenAIStreamStarted)
        await stream.aclose()

        self.assertEqual(handle.cancelled, ["stream-1"])


if __name__ == "__main__":
    unittest.main()
