from __future__ import annotations

import json
import pathlib
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))

from meshllm import Client, OpenAIRequestError, RequestCompleted, TextDelta


class FakeEvent:
    def __init__(self, kind: str, request_id: str, value: str | None = None) -> None:
        self.kind = kind
        self.request_id = request_id
        self.delta = value
        self.error = value

    def is_token_delta(self) -> bool:
        return self.kind == "delta"

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
        return [SimpleNamespace(id="model-a", name="Model A")]

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

    def chat(self, request: object, listener: object) -> str:
        listener.on_event(FakeEvent("delta", "req-1", "hello"))
        listener.on_event(FakeEvent("completed", "req-1"))
        return "req-1"

    def cancel(self, request_id: str) -> None:
        self.cancelled.append(request_id)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_lifecycle_and_models(self) -> None:
        handle = FakeHandle()
        client = Client(handle)

        async with client:
            self.assertTrue((await client.status()).connected)
            self.assertEqual((await client.inference.list_models())[0].id, "model-a")

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
        })

        self.assertEqual(handle.last_openai[0], "/v1/chat/completions")
        self.assertEqual(handle.last_openai[1]["tools"], [tool])
        self.assertEqual(result["choices"][0]["message"]["tool_calls"], [tool])
        self.assertEqual(result["usage"]["total_tokens"], 12)

    async def test_non_success_response_raises_typed_error(self) -> None:
        client = Client(FakeHandle())

        with self.assertRaises(OpenAIRequestError) as raised:
            await client.inference.chat_completions({"model": "missing", "messages": []})

        self.assertEqual(raised.exception.status_code, 404)

    async def test_text_stream_is_async_and_cancels_native_request(self) -> None:
        from meshllm import client as client_module

        original_native = client_module.native
        client_module.native = lambda: SimpleNamespace(
            ChatRequestNative=lambda **kwargs: SimpleNamespace(**kwargs),
            ChatMessageNative=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        handle = FakeHandle()
        try:
            events = [
                event
                async for event in Client(handle).inference.chat(
                    model="model-a", messages=[{"role": "user", "content": "hello"}]
                )
            ]
        finally:
            client_module.native = original_native

        self.assertEqual(events, [TextDelta("req-1", "hello"), RequestCompleted("req-1")])
        self.assertEqual(handle.cancelled, [])


if __name__ == "__main__":
    unittest.main()
