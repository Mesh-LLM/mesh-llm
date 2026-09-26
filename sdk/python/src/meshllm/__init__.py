from __future__ import annotations

from ._binding import native
from .client import Client, Inference, Node
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

__all__ = [
    "Client",
    "Inference",
    "InferenceEvent",
    "MeshError",
    "Model",
    "Node",
    "OpenAIRequestError",
    "OpenAIResponse",
    "RequestCompleted",
    "Status",
    "TextDelta",
    "generate_owner_keypair_hex",
]

__version__ = "0.76.1"


def generate_owner_keypair_hex() -> str:
    return native().generate_owner_keypair_hex()
