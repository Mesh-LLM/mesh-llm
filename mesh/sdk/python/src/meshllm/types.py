from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class MeshError(RuntimeError):
    """Base exception raised by the Python SDK."""


class OpenAIRequestError(MeshError):
    def __init__(
        self,
        status_code: int | None,
        body: str | None,
        *,
        message: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.body = body
        self.message = message
        detail = message or body or "OpenAI-compatible request failed"
        if status_code is None:
            super().__init__(detail)
        else:
            super().__init__(f"OpenAI-compatible request failed with HTTP {status_code}: {detail}")


@dataclass(frozen=True, slots=True)
class Model:
    id: str
    name: str
    context_length: int | None = None


@dataclass(frozen=True, slots=True)
class Status:
    running: bool
    mode: str
    api_base_url: str
    console_url: str
    payload: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class OpenAIStreamStarted:
    request_id: str
    status_code: int
    content_type: str | None


@dataclass(frozen=True, slots=True)
class OpenAIStreamChunk:
    request_id: str
    event: str | None
    data: str
    raw: str

    @property
    def done(self) -> bool:
        return self.data == "[DONE]"

    def json(self) -> Any:
        if self.done:
            return None
        return json.loads(self.data)


OpenAIStreamEvent = OpenAIStreamStarted | OpenAIStreamChunk


@dataclass(frozen=True, slots=True)
class OpenAIResponse:
    status_code: int
    content_type: str | None
    body: str

    def json(self) -> dict[str, Any]:
        value = json.loads(self.body)
        if not isinstance(value, dict):
            raise MeshError("OpenAI-compatible response body was not a JSON object")
        return value
