from __future__ import annotations

from ._binding import native
from .client import Client, Inference, Node
from .types import (
    MeshError,
    Model,
    OpenAIRequestError,
    OpenAIResponse,
    OpenAIStreamChunk,
    OpenAIStreamEvent,
    OpenAIStreamStarted,
    Status,
)

__all__ = [
    "Client",
    "Inference",
    "MeshError",
    "Model",
    "Node",
    "OpenAIRequestError",
    "OpenAIResponse",
    "OpenAIStreamChunk",
    "OpenAIStreamEvent",
    "OpenAIStreamStarted",
    "Status",
    "generate_owner_keypair_hex",
]

__version__ = "0.76.1"


def generate_owner_keypair_hex() -> str:
    return native().generate_owner_keypair_hex()
