#!/usr/bin/env python3
"""Official openai-python embeddings smoke against a compatible endpoint."""

from __future__ import annotations

import argparse
import base64
import math
import struct

from workload_fixtures import EMBEDDING_INPUTS


def main() -> None:
    """Check batched numeric and base64 vectors through the official Python SDK."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "openai package not installed; run `python -m pip install openai` first"
        ) from exc

    client = OpenAI(api_key="mesh-llm-ci", base_url=args.base_url)
    inputs = list(EMBEDDING_INPUTS)
    response = client.embeddings.create(
        model=args.model,
        input=inputs,
        encoding_format="float",
    )
    if response.object != "list" or response.model != args.model:
        raise RuntimeError("embeddings response has the wrong object or model")
    if len(response.data) != len(inputs):
        raise RuntimeError("embeddings response has the wrong batch size")
    dimensions: int | None = None
    for index, item in enumerate(response.data):
        if item.object != "embedding" or item.index != index:
            raise RuntimeError("embeddings response has invalid item metadata")
        if not item.embedding or not all(math.isfinite(value) for value in item.embedding):
            raise RuntimeError("embeddings response contains no finite vector")
        norm = math.sqrt(sum(value * value for value in item.embedding))
        if abs(norm - 1.0) > 1e-4:
            raise RuntimeError(f"embedding {index} is not L2-normalized: {norm}")
        if dimensions is None:
            dimensions = len(item.embedding)
        elif dimensions != len(item.embedding):
            raise RuntimeError("embedding dimensions differ within one response")
    if response.usage.prompt_tokens <= 0:
        raise RuntimeError("embeddings usage reports no prompt tokens")

    encoded = client.embeddings.create(
        model=args.model,
        input=inputs[0],
        encoding_format="base64",
    )
    payload = encoded.data[0].embedding
    if not isinstance(payload, str):
        raise RuntimeError("base64 embedding did not deserialize as a string")
    raw = base64.b64decode(payload, validate=True)
    if dimensions is None or len(raw) != dimensions * struct.calcsize("<f"):
        raise RuntimeError("base64 embedding has the wrong byte length")
    values = struct.unpack(f"<{dimensions}f", raw)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("base64 embedding contains a non-finite value")

    print(
        f"openai-python embeddings smoke passed: model={args.model} "
        f"batch={len(inputs)} dimensions={dimensions}"
    )


if __name__ == "__main__":
    main()
