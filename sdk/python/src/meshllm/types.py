from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class MeshError(RuntimeError):
    """Base exception raised by the Python SDK."""


class OpenAIRequestError(MeshError):
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"OpenAI-compatible request failed with HTTP {status_code}: {body}")


@dataclass(frozen=True, slots=True)
class Model:
    id: str
    name: str


@dataclass(frozen=True, slots=True)
class Status:
    connected: bool
    peer_count: int


@dataclass(frozen=True, slots=True)
class TextDelta:
    request_id: str
    text: str


@dataclass(frozen=True, slots=True)
class RequestCompleted:
    request_id: str


InferenceEvent = TextDelta | RequestCompleted


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
